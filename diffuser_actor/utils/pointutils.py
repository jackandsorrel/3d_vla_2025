import dgl.geometry as dgl_geo
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def fps(points, number):
    """
    Farthest Point Sampling (FPS) for point clouds.
    
    Args:
        points: Input point cloud of shape (bs, num_points, 3)
        number: Number of points to sample
    
    Returns:
        sampled_points: Sampled points of shape (bs, number, 3)
    """
    bs, num_points, _ = points.shape

    # Sample indices using FPS
    sampled_inds = dgl_geo.farthest_point_sampler(
        points.to(torch.float64),  # Convert to float64 for stability
        number, 
        0  # Start index
    ).long()
    
    # Gather the sampled points
    expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, 3)
    sampled_points = torch.gather(
        points, 
        1, 
        expanded_sampled_inds
    )
    
    return sampled_points

def square_distance(src, dst):
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

def query_knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :k]
    return group_idx

def index_points(points, idx, is_group=False):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, theta, phi) [B, N, 3] / [B, N, G, 3]
    """
    # rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    # rho = torch.clamp(rho, min=0)  # range: [0, inf]
    # theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    # phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # # check nan
    # idx = rho == 0
    # theta[idx] = 0

    # if normalize:
    #     theta = theta / np.pi  # [0, 1]
    #     phi = phi / (2 * np.pi) + .5  # [0, 1]
    # out = torch.cat([rho, theta, phi], dim=-1)
    # return out
    # Small epsilon value for numerical stability
    eps = 1e-10
    safe_xyz = xyz.clone()  # Avoid modifying input tensor
    # assert not torch.isnan(safe_xyz).any(), "Input contains NaN!"
    # assert not torch.isinf(safe_xyz).any(), "Input contains Inf!"
    
    # Compute radius with protection
    squared_sum = torch.sum(torch.pow(safe_xyz, 2), dim=-1, keepdim=True)
    rho = torch.sqrt(squared_sum + eps)  # Add epsilon before sqrt
    
    # Compute theta (polar angle) with protection
    z_div_rho = safe_xyz[..., 2:3] / torch.clamp(rho, min=eps)
    theta = torch.acos(torch.clamp(z_div_rho, min=-1.0+eps, max=1.0-eps))
    
    # Compute phi (azimuthal angle) with protection
    # Add epsilon only to near-zero values
    x = safe_xyz[..., 0:1] + eps * (safe_xyz[..., 0:1].abs() < eps)
    y = safe_xyz[..., 1:2] + eps * (safe_xyz[..., 1:2].abs() < eps)
    phi = torch.atan2(y, x)
    
    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + 0.5  # [0, 1]
    
    # Final output with NaN check
    result = torch.cat([rho, theta, phi], dim=-1)
    if torch.isnan(result).any():
        print("Warning: NaN detected in spherical conversion")
        result = torch.nan_to_num(result, nan=0.0)
    
    return result

def resort_points(points, idx):
    """
    Resort Set of points along G dim

    """
    device = points.device
    B, N, G, _ = points.shape

    view_shape = [B, 1, 1]
    repeat_shape = [1, N, G]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points


def group_by_umbrella(xyz, new_xyz, k=9):
    """
    Group a set of points into umbrella surfaces

    """
    idx = query_knn_point(k, xyz, new_xyz)
    torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
    torch.cuda.empty_cache()

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N', K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1]

    # [B, N', K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz

def cal_normal(group_xyz, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    :param group_xyz: [B, N, K=3, 3] / [B, N, G, K=3, 3]
    :param random_inv:
    :param return_intersect:
    :param return_const:
    :return: [B, N, 3]
    """
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [B, N, 3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [B, N, 3]

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
    # unit_nor = nor / torch.norm(nor, dim=-1, keepdim=True)  # [B, N, 3] / [B, N, G, 3]

    norm = torch.norm(nor, dim=-1, keepdim=True)
    safe_norm = torch.where(norm < 1e-6, torch.ones_like(norm), norm)  # Replace zeros with 1
    unit_nor = nor / safe_norm

    if not is_group:
        pos_mask = (unit_nor[..., 0] > 0).float() * 2. - 1.  # keep x_n positive
    else:
        pos_mask = (unit_nor[..., 0:1, 0] > 0).float() * 2. - 1.
    unit_nor = unit_nor * pos_mask.unsqueeze(-1)

    # batch-wise random inverse normal vector (prob: 0.5)
    if random_inv:
        random_mask = torch.randint(0, 2, (group_xyz.size(0), 1, 1)).float() * 2. - 1.
        random_mask = random_mask.to(unit_nor.device)
        if not is_group:
            unit_nor = unit_nor * random_mask
        else:
            unit_nor = unit_nor * random_mask.unsqueeze(-1)

    return unit_nor

def cal_center(group_xyz):
    """
    Calculate Global Coordinates of the Center of Triangle

    :param group_xyz: [B, N, K, 3] / [B, N, G, K, 3]; K >= 3
    :return: [B, N, 3] / [B, N, G, 3]
    """
    center = torch.mean(group_xyz, dim=-2)
    return center

def cal_const(normal, center, is_normalize=True):
    """
    Calculate Constant Term (Standard Version, with x_normal to be 1)

    math::
        const = x_nor * x_0 + y_nor * y_0 + z_nor * z_0

    :param is_normalize:
    :param normal: [B, N, 3] / [B, N, G, 3]
    :param center: [B, N, 3] / [B, N, G, 3]
    :return: [B, N, 1] / [B, N, G, 1]
    """
    # const = torch.sum(normal * center, dim=-1, keepdim=True)
    # factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
    # const = const / factor if is_normalize else const

    # return const
    eps = 1e-6
    max_val = 1e6
    
    # # 1. Input validation
    # assert torch.isfinite(normal).all(), "Normal vectors contain NaN/Inf"
    # assert torch.isfinite(center).all(), "Center points contain NaN/Inf"
    
    # 2. Normal vector sanitization
    normal_norm = torch.norm(normal, dim=-1, keepdim=True)
    safe_normal = torch.where(normal_norm < eps,
                           torch.tensor([1., 0., 0.], device=normal.device),
                           normal / torch.clamp(normal_norm, min=eps))
    
    # 3. Center point stabilization
    safe_center = torch.clamp(center, -max_val, max_val)
    
    # 4. Safe dot product computation
    dot_product = torch.sum(safe_normal * safe_center, dim=-1, keepdim=True)
    
    # 5. Optional normalization with protection
    if is_normalize:
        factor = torch.sqrt(torch.tensor(3.0, device=dot_product.device)) + eps
        const = dot_product / factor
    else:
        const = dot_product
    
    # 6. Final validation
    if torch.isnan(const).any() or torch.isinf(const).any():
        print(f"Warning: Invalid const values detected - min: {const.min()}, max: {const.max()}")
        const = torch.nan_to_num(const, nan=0.0, posinf=max_val, neginf=-max_val)
    
    return const

def check_nan_umb(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    :param pos: [B, N, G, 1]
    :param center: [B, N, G, 3]
    :param normal: [B, N, G, 3]
    :return:
    """
    B, N, G, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    mask_first = torch.argmax((~mask).int(), dim=-1)
    b_idx = torch.arange(B).unsqueeze(1).repeat([1, N])
    n_idx = torch.arange(N).unsqueeze(0).repeat([B, 1])

    normal_first = normal[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
    normal[mask] = normal_first[mask]
    center_first = center[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center

class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella-based Surface Abstraction Module

    """

    def __init__(self, k, in_channel, aggr_type='sum', return_dist=False, random_inv=True):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type = aggr_type

        print(f"In channel: {in_channel}")

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        )

    def forward(self, center):
        """
        center: [B, N, 3]
        """
        # center = center.permute(0, 2, 1)
        # surface construction

        group_xyz = group_by_umbrella(center, center, k=self.k)  # [B, N, K-1, 3 (points), 3 (coord.)]
        # 在调用 cal_normal 前添加检查
        # assert torch.isfinite(group_xyz).all(), "group_xyz contains NaN/Inf"

        # normal
        group_normal = cal_normal(group_xyz, random_inv=self.random_inv, is_group=True)
        # assert torch.isfinite(group_normal).all(), f"group_normal has NaN/Inf. Stats: {group_normal.mean()}, {group_normal.std()}"
        # coordinate
        group_center = cal_center(group_xyz)
        # assert torch.isfinite(group_center).all(), "group_center has NaN/Inf"
        # polar
        group_polar = xyz2sphere(group_center)
        # assert torch.isfinite(group_polar).all(), "group_polar has NaN/Inf"
        if self.return_dist:
            group_pos = cal_const(group_normal, group_center)
            group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
            new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        else:
            group_normal, group_center = check_nan_umb(group_normal, group_center)
            new_feature = torch.cat([group_center, group_polar, group_normal], dim=-1)
        new_feature = new_feature.permute(0, 3, 2, 1)  # [B, C, G, N]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        if self.aggr_type == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]

    FLOPs:
        S * (3 + 3 + 2)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def sample_and_group(npoint, nsample, center, normal, feature, return_normal=True, return_polar=False):
    """
    Input:
        center: input points position data
        normal: input points normal data
        feature: input points feature
    Return:
        new_center: sampled points position data
        new_normal: sampled points normal data
        new_feature: sampled points feature
    """
    # sample
    fps_idx = farthest_point_sample(center, npoint)  # [B, npoint, A]
    torch.cuda.empty_cache()
    # sample center
    new_center = index_points(center, fps_idx, is_group=False)
    torch.cuda.empty_cache()
    # sample normal
    new_normal = index_points(normal, fps_idx, is_group=False)
    torch.cuda.empty_cache()

    # group
    # idx = query_ball_point(radius, nsample, center, new_center, cuda=cuda)
    idx = query_knn_point(nsample, center, new_center)
    torch.cuda.empty_cache()
    # group normal
    group_normal = index_points(normal, idx, is_group=True)  # [B, npoint, nsample, B]
    torch.cuda.empty_cache()
    # group center
    group_center = index_points(center, idx, is_group=True)  # [B, npoint, nsample, A]
    torch.cuda.empty_cache()
    group_center_norm = group_center - new_center.unsqueeze(2)
    torch.cuda.empty_cache()

    # group polar
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        # assert torch.isfinite(group_polar).all(), "group_polar has NaN/Inf"
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)
    if feature is not None:
        group_feature = index_points(feature, idx, is_group=True)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1) if return_normal \
            else torch.cat([group_center_norm, group_feature], dim=-1)
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1)

    return new_center, new_normal, new_feature

class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, nsample, feat_channel, pos_channel, mlp, embedding_dim,
                 return_normal=True, return_polar=True):
        super(SurfaceAbstractionCD, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel

        self.mlp_l0 = nn.Conv2d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv2d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm2d(mlp[0])
        self.bn_f0 = nn.BatchNorm2d(mlp[0])

        # mlp_l0+mlp_f0 can be considered as the first layer of mlp_convs
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.projection = nn.Linear(last_channel, embedding_dim)

    def forward(self, center, normal, feature):
        # normal = normal.permute(0, 2, 1)
        # center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        new_center, new_normal, new_feature = sample_and_group(self.npoint, self.nsample, center,
                                                                normal, feature, return_normal=self.return_normal,
                                                                return_polar=self.return_polar)

        new_feature = new_feature.permute(0, 3, 2, 1)

        # init layer
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
        new_feature = loc + feat
        new_feature = F.relu(new_feature)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0].permute(0, 2, 1)

        new_feature = self.projection(new_feature)

        # new_center = new_center.permute(0, 2, 1)
        # new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature