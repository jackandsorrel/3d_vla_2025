import dgl.geometry as dgl_geo
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork
from torch.utils.checkpoint import checkpoint
import math
import MinkowskiEngine as ME

from .position_encodings import RotaryPositionEncoding3D
from .layers import FFWRelativeCrossAttentionModule, ParallelAttention
from .resnet import load_resnet50, load_resnet18
from .clip import load_clip
from .constants import WORKSPACE
from .pointutils import fps, UmbrellaSurfaceConstructor, SurfaceAbstractionCD, group_by_umbrella, cal_normal
from .minkowski.resnet import ResNet14

class Group(nn.Module):
    def __init__(self, group_config, use_color = False):
        """
        Args:
            group_configs: List of (num_group, group_size) tuples for each level
                          Example: [(10000, 1), (512, 64)] for 2-level hierarchy
            use_color: Whether to use color features
        """
        super().__init__()
        self.group_config = group_config
        self.use_color = use_color
        print(f"Clustering group config: {group_config}")
        print(f"Use_color: {use_color}")
        
    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm;
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist 
    
    def knn_point(self, nsample, xyz, new_xyz):
        """
        Input:
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        sqrdists = self.square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
        return group_idx
    
    def forward(self, xyz, color = None):
        """
        Input:
            xyz: (B, N, 3) point coordinates
            color: (B, N, 3) optional color features
        Output:
            neighborhoods: List of (B, G, M, 3) tensors per level
            centers: List of (B, G, 3) tensors per level
            features: List of (B, G, M, 6) tensors per level (if use_color)
        """

        for i, (num_group, group_size) in enumerate(self.group_config):
            batch_size, num_points, _ = xyz.shape

            # fps the centers out
            center = fps(xyz, num_group) # B G 3

            if group_size > 1:
                # knn to get the neighborhood
                idx = self.knn_point(group_size, xyz, center) # B G M
                idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
                idx = idx + idx_base
                idx = idx.view(-1)

                # gather neighborhood
                neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
                neighborhood = neighborhood.view(batch_size, num_group, group_size, 3).contiguous()

                # normalize
                neighborhood = neighborhood - center.unsqueeze(2)

                if color is not None and self.use_color:
                    neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
                    neighborhood_color = neighborhood_color.view(batch_size, num_group, group_size, 3).contiguous()

                    features = torch.cat((neighborhood, neighborhood_color), dim=-1)
                else:
                    features = None
            
            xyz = center

            if color is not None:
                # For color propagation, use colors at the sampled center points
                color = self.knn_point(1, color, center).squeeze(2)
            
        return neighborhood, center, features

    def match_embeddings(self, neighborhood, centers, rgb_embeddings, patch_pcds):
        """
        Args:
            neighborhood: (B, G, E) - Cluster point clouds embedding
            centers: (B, G, 3) - Cluster centers
            rgb_embeddings: (B, P, D) - RGB patch embeddings (P=4096, D=60)
            patch_pcds: (B, P, 3) - 3D positions of RGB patches
            
        Returns:
            matched_embeddings: (B, P, E) - Nearest cluster embedding for each patch
            patch_distances: (B, P) - Distance from each patch to nearest cluster
        """
        B, G, _ = neighborhood.shape
        P = rgb_embeddings.shape[1]
        
        dist_matrix = torch.cdist(patch_pcds, centers)  # (B, P, G)
        
        min_dist, nearest_cluster_idx = dist_matrix.min(dim=-1)  # (B, P)
        
        batch_indices = torch.arange(B, device=neighborhood.device)[:, None].expand(B, P)
        matched_embeddings = neighborhood[batch_indices, nearest_cluster_idx]  # (B, P, E)
        
        return matched_embeddings, min_dist


class Conv(nn.Module):
    """
    Pointcloud conv function to encode xyz data.

    Inputs:
    point_groups: (B, patch_num, N, 3)
    color_groups: (B, patch_num, N, 3)

    Outputs:
    point_embeddings: (B, patch_num, embedding_dim)
    """
    
    def __init__(self, input_dim=3, output_dim=60, bn_momentum=0.02):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.cloud_encoder = ResNet14(
            in_channels=input_dim,
            out_channels=output_dim,
            conv1_kernel_size=3,
            dilations=(1, 1, 1, 1),
            bn_momentum=bn_momentum,
            D=3
        )
    
    def forward(self, point_groups, color_groups=None):
        B, patch_num, N, _ = point_groups.shape
        
        if color_groups is not None:
            features = torch.cat([point_groups, color_groups], dim=-1)
        else:
            features = point_groups
        
        features_flat = features.view(B * patch_num, N, self.input_dim)
        coords_flat = point_groups.view(B * patch_num, N, 3)
        
        coords_list = []
        feats_list = []
        
        for i in range(B * patch_num):
            patch_coords = coords_flat[i] # (N, 3)
            patch_features = features_flat[i]  # (N, input_dim)
            
            coords_list.append(patch_coords)
            feats_list.append(patch_features)
        
        coords_batch, feats_batch = ME.utils.sparse_collate(
            coords_list, feats_list, dtype=torch.float32
        )
        
        sparse_tensor = ME.SparseTensor(
            features=feats_batch,
            coordinates=coords_batch,
        )
        
        encoded = self.cloud_encoder(sparse_tensor)
        print(f"Encoded shape: {encoded.shape}")
        
        embeddings = torch.zeros([B * patch_num, self.output_dim],
                               device=encoded.F.device, dtype=encoded.F.dtype)
        
        batch_indices = encoded.C[:, 0].long()
        
        for batch_idx in range(B * patch_num):
            mask = (batch_indices == batch_idx)
            if mask.any():
                batch_features = encoded.F[mask]
                embeddings[batch_idx] = batch_features.max(dim=0)[0]
        
        print(f"Embeddings shape: {embeddings.shape}")

        # 重塑回原始形状
        return embeddings.view(B, patch_num, self.output_dim)



class PointNet(nn.Module):
    """
    Small pointnet to encode xyz data.

    Inputs:
    point_groups: (B, patch_num, N, 3)
    color_groups: (B, patch_num, N, 3)

    Outputs:
    point_embeddings: (B, patch_num, embedding_dim)
    centers: (B, patch_num, 3) to use for positional embeddings
    """
    def __init__(self, embedding_dim, use_colors = True, batch_norm = False):
        super().__init__()
        self.use_colors = use_colors
        print(f"Use colors: {use_colors}, Batch Norm: {batch_norm}")
        self.in_channels = 6 if use_colors else 3
        print(f"In channels: {self.in_channels}")

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, 64),
            nn.LayerNorm(64) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256) if batch_norm else nn.Identity(),
            nn.ReLU(),
            *([
                nn.Linear(256, 512),
                nn.LayerNorm(512) if batch_norm else nn.Identity(),
                nn.ReLU(),
            ] if use_colors else [])
        )
        # self.final_projection = nn.Linear(self.in_channels, embedding_dim)
        self.final_projection = nn.Linear(512 if use_colors else 256, embedding_dim)
    
    def forward(self, point_groups = None, color_groups = None, inference = False):
        if color_groups is not None and self.use_colors:
            x = torch.cat([point_groups, color_groups], dim=-1)  # (B, patch_num, N, 6)
        else:
            x = point_groups  # (B, group_num, N, 3)

        B, patch_num, N, _ = point_groups.shape
        x = x.view(B * patch_num, N, self.in_channels)  # (B * group_num, N, in_channels)

        x = self.mlp(x)  # (B * patch_num, N, 512) 或 (B * patch_num, N, 256)
        max_features = torch.max(x, 1)[0]  # (B * patch_num, 512) 或 (B * patch_num, 256)
        max_features = self.final_projection(max_features)  # (B * patch_num, out_channels)
        max_features = max_features.view(B, patch_num, -1)  # (B, patch_num, out_channels)
        return max_features


class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 backbone_dir=None,
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_sampling_level=3,
                 nhist=3,
                 num_attn_heads=8,
                 num_vis_ins_attn_layers=2,
                 fps_subsampling_factor=5,
                 use_colors=False,
                 pcd_method=None,
                 group_config=None,
                 filter_points=False,
                 num_pcd_attn_layers=2,
                 use_color=False,
                 use_repsurf=False,
                 return_dist=False,
                 downsample_points=True,
                 pcd_backbone="pointnet",
                 ):
        super().__init__()
        assert backbone in ["resnet50", "resnet18", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.fps_subsampling_factor = fps_subsampling_factor
        self.filter_points = filter_points
        self.use_repsurf = use_repsurf
        self.downsample_points = downsample_points
        self.pcd_backbone = pcd_backbone
        # self.use_color = use_color

        # Frozen backbone
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip(backbone_dir=backbone_dir)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], embedding_dim
        )
        self.pointnet = PointNet(embedding_dim=embedding_dim, use_colors=use_color)
        
        self.pcd_method = pcd_method

        self.sample_num = group_config[0][0]
        print(f"sample num:{self.sample_num}")

        if return_dist:
            repsurf_channel = 10
        else:
            repsurf_channel = 9
        
        center_channel = 6
        
        if use_repsurf:
            if return_dist:
                self.umbrella_module = UmbrellaSurfaceConstructor(k=9, in_channel=10, return_dist=True)
            else:
                self.umbrella_module = UmbrellaSurfaceConstructor(9, 9)
            self.sa = SurfaceAbstractionCD(npoint=group_config[1][0], 
                                           nsample=group_config[1][1], 
                                           feat_channel=repsurf_channel, 
                                           pos_channel=center_channel, 
                                           mlp=[64, 64, 128], 
                                           embedding_dim=embedding_dim)

        if "cluster" in self.pcd_method:
            self.group_module = Group(group_config=group_config, use_color=use_color) # group clustering module
        elif "patch" in self.pcd_method:
            if "conv" in self.pcd_backbone:
                self.conv = Conv(input_dim=3, output_dim=embedding_dim)

        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid
            # at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid
            # at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Instruction encoder
        self.instruction_encoder = nn.Linear(512, embedding_dim)

        # Attention from vision to language
        layer = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=embedding_dim, n_heads=num_attn_heads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.vl_attention = nn.ModuleList([
            layer
            for _ in range(1)
            for _ in range(1)
        ])
        self.pcd_attn = ParallelAttention(
            num_layers=num_pcd_attn_layers,
            d_model=embedding_dim,
            n_heads=num_attn_heads,
            self_attention1=False,
            self_attention2=False,
            cross_attention1=True,
            cross_attention2=False
        )
        

    def forward(self):
        return None

    def encode_curr_gripper(self, curr_gripper, context_feats, context):
        """
        Compute current gripper position features and positional embeddings.

        Args:
            - curr_gripper: (B, nhist, 3+)

        Returns:
            - curr_gripper_feats: (B, nhist, F)
            - curr_gripper_pos: (B, nhist, F, 2)
        """
        return self._encode_gripper(curr_gripper, self.curr_gripper_embed,
                                    context_feats, context)

    def encode_goal_gripper(self, goal_gripper, context_feats, context):
        """
        Compute goal gripper position features and positional embeddings.

        Args:
            - goal_gripper: (B, 3+)

        Returns:
            - goal_gripper_feats: (B, 1, F)
            - goal_gripper_pos: (B, 1, F, 2)
        """
        goal_gripper_feats, goal_gripper_pos = self._encode_gripper(
            goal_gripper[:, None], self.goal_gripper_embed,
            context_feats, context
        )
        return goal_gripper_feats, goal_gripper_pos

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, npt, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: (B, npt, F)
            - gripper_pos: (B, npt, F, 2)
        """
        # Learnable embedding for gripper
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(
            gripper_feats, 'b npt c -> npt b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(
            gripper_feats, 'nhist b c -> b nhist c'
        )

        return gripper_feats, gripper_pos
    
    def imagetopatch(self, images=None, patch_size=None):
        '''
        Turn rgb & depth images to patches.
        Input: (frame_ids, C, H, W)  if single view
        Output: (frame_ids, patch_num, H_p, W_p, C)
        '''
        # Ensure input has shape (F, C, H, W)
        F, C, H, W = images.shape
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"Image dimensions must be divisible by patch size. Got H={H}, W={W}, patch_size={patch_size}"
        
        # Calculate number of patches
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        patch_num = num_patches_h * num_patches_w

        # Unfold the image into patches
        # Reorder dimensions to (F, num_patches_h, num_patches_w, C, H_p, W_p)
        patches = images.unfold(2, patch_size, patch_size). \
                            unfold(3, patch_size, patch_size). \
                            permute(0, 2, 3, 4, 5, 1). \
                            reshape(F, patch_num, patch_size, patch_size, C)  # Reshape to (F, patch_num, H_p, W_p, C)
        
        return patches

    def encode_images(self, rgb, pcd, inference = False):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        # points = einops.rearrange(pcd, 'b cam c h w -> b (cam h w) c')
        # bs, _, c = points.shape
        # device = points.device

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        pcd_pyramid = []
        
        for i in range(self.num_sampling_level):
            # Isolate level's visual features
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            feat_h, feat_w = rgb_features_i.shape[-2:]

            pcd_i = F.interpolate(
                pcd,
                (feat_h, feat_w),
                mode='bilinear'
            )

            # Merge different cameras for clouds, separate for rgb features
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(
                pcd_i,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i,
                "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_features_i = einops.rearrange(
                rgb_features_i,
                "b ncam c h w -> b (ncam h w) c"
            )
            
            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid
    
    def sample(self, points, sample_num, normalize = True):
        # points = points.unsqueeze(0)  # (1, N, C)

        # sampled_points = fps(points, sample_num)  # (sample_num, c)
        sampled_points = points[:sample_num]
        
        if normalize:
            # Center points
            centroid = sampled_points.mean(dim=0)
            centered = sampled_points - centroid
            
            # Scale to [-1, 1]
            max_dist = centered.abs().max()
            if max_dist > 1e-6:  # Avoid division by zero
                centered /= max_dist
            
            return centered
        
        return sampled_points

    def downsample(self, points, sample_num):
        """
        Inputs:
            points: (bs cam c h w)
        Return:
            downsampled_points: (bs, cam, c, h_new, w_new)
        """
        side_length = int(math.isqrt(sample_num))  # 例如 100x100=10,000
        
        points = points.squeeze(1)  # (bs, 1, 3, 256, 256) -> (bs, 3, 256, 256)
        downsampled = F.interpolate(
            points,
            size=(side_length, side_length),
            mode='bilinear',  # 对 XYZ 坐标插值
            align_corners=False
        )
        downsampled = downsampled.unsqueeze(1)  # (bs, 3, h_new, w_new) -> (bs, 1, 3, h_new, w_new)
        
        return downsampled 
    
    def get_rotation(self, points, pcd_centers, k=9, random_inv=False):
        """
        Get center rotation from repsurf method.
        Input:
            points: (bs cam c h w) - point features
            pcd_centers: (bs N 3)
        Return:
            rot: (bs N C) - rotation for patch centers
        """

        bs, cam, c, h, w = points.shape
        N = pcd_centers.size(1)
        
        xyz = points
        xyz = xyz.permute(0, 1, 3, 4, 2)  # [bs, cam, h, w, 3]
        xyz = xyz.reshape(bs, cam * h * w, 3)  # [bs, cam*h*w, 3]
        
        umbrella_groups = group_by_umbrella(xyz, pcd_centers, k=k)
        # [bs, N, K-1, 3, 3]
        
        normals = cal_normal(umbrella_groups, random_inv=random_inv, is_group=True)
        # [bs, N, K, 3]

        normals = torch.mean(normals, dim=2)  # [bs, N, 3]
        
        return normals
        
    
    def encode_pcds(self, points, context_feats, context):
        """
        Cluster point clouds and match rgb context features to the closest point cloud cluster features.
        Input:
            points: (bs cam c h w) - point features
            context_feats: (bs patch_num )
        Return:
            pcd_feats: (bs, cam_num * 32 * 32, dim) - point cloud features
        """
        device = points.device
        if self.downsample_points:
            points = self.downsample(points, self.sample_num)
        if "cluster" in self.pcd_method:
            points = einops.rearrange(points, 'bs cam c h w -> bs (cam h w) c')
            bs, _, c = points.shape
            
            if self.use_repsurf:
                # torch.autograd.set_detect_anomaly(True)
                # TODO: finish adding repsurf in here
                points = points.requires_grad_(True) 
                points_feature = self.umbrella_module(points) # shape of points feature: [B, C, N]
                points_feature = points_feature.permute(0, 2, 1)
                center, normal, neighborhood_embed = self.sa(points, points_feature, None)
            else:
                points = points.requires_grad_(True) 
                neighborhood, center, _ = self.group_module(points) # torch.Size([6, 256, 3])
                neighborhood_embed = self.pointnet(neighborhood) # torch.Size([6, 256, 60])

            matched_embed, min_dist = self.group_module.match_embeddings(neighborhood_embed, center, context_feats, context)  # matched_embed (bs, P, embedding_dim)
            valid_mask = (min_dist < 1.0)  # (B, P)

            return matched_embed, valid_mask
        
        elif "patch" in self.pcd_method:
            # TODO: fix this
            bs = points.shape[0]
            points = einops.rearrange(points, 'bs ncam C H W -> (bs ncam) C H W')
            # print(f"Points shape: {points.shape}")
            pcd_patches = self.imagetopatch(images=points, patch_size=int(self.image_size[0] / 32)) # shape [(bt ncam) patch_num H W c]
            pcd_patches = einops.rearrange(
                pcd_patches,
                "bt patch_num H W c -> bt patch_num (H W) c"
            )
            # print(f"Patches shape:{pcd_patches.shape}")
            if "pointnet" in self.pcd_backbone:
                pcd_feats = self.pointnet(pcd_patches)  # torch.size([24, 32 * 32, 60])
                matched_embed = einops.rearrange(pcd_feats, '(bs ncam) P dim -> bs (ncam P) dim', bs=bs)
            elif "conv" in self.pcd_backbone:
                pcd_feats = self.conv(pcd_patches)
                matched_embed = einops.rearrange(pcd_feats, '(bs ncam) P dim -> bs (ncam P) dim', bs=bs)

            return matched_embed, None



    def encode_instruction(self, instruction):
        """
        Compute language features/pos embeddings on top of CLIP features.

        Args:
            - instruction: (B, max_instruction_length, 512)

        Returns:
            - instr_feats: (B, 53, F)
            - instr_dummy_pos: (B, 53, F, 2)
        """
        instr_feats = self.instruction_encoder(instruction)
        # Dummy positional embeddings, all 0s
        instr_dummy_pos = torch.zeros(
            len(instruction), instr_feats.shape[1], 3,
            device=instruction.device
        )
        instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        return instr_feats, instr_dummy_pos

    def run_fps(self, context_features, context_pos):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(
                context_features,
                "npts b c -> b npts c"
            ).to(torch.float64),
            max(npts // self.fps_subsampling_factor, 1), 0
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )

        # Sample positional embeddings
        _, _, ch, npos = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        )
        sampled_context_pos = torch.gather(
            context_pos, 1, expanded_sampled_inds
        )
        return sampled_context_features, sampled_context_pos

    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats
    
    def pcd_rgb_attention(self, context_feats, pcd_feats):
        context_feats, _ = self.pcd_attn(
            seq1=context_feats, 
            seq1_key_padding_mask=None,
            seq2=pcd_feats,
            seq2_key_padding_mask=None,
            seq1_pos=None,
            seq2_pos=None,
            seq1_sem_pos=None,
            seq2_sem_pos=None
        )
        return context_feats
    
