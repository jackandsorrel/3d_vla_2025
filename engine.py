"""Shared utilities for all main scripts."""

import os
import pickle
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):
        """Initialize."""
        self.args = args

        if (bool(self.args.distributed) and dist.get_rank() == 0) or not bool(self.args.distributed):
            args.save(str(args.log_dir / "hparams.json"))

        self.args = args

        if (bool(self.args.distributed) and dist.get_rank() == 0) or not bool(self.args.distributed):
            self.writer = SummaryWriter(log_dir=args.log_dir)

    @staticmethod
    def get_datasets():
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset

    def get_loaders(self, collate_fn=default_collate):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        # Datasets
        train_dataset, test_dataset = self.get_datasets()
        # print("Full dataset size (before DistributedSampler):", len(train_dataset)) 
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        # TODO: change this
        if (bool(self.args.distributed)):
            train_sampler = DistributedSampler(train_dataset)
            # print(f"Rank {torch.distributed.get_rank()}: Sampler indices = {list(train_sampler)}")
        else:
            train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g
        )
        if (bool(self.args.distributed)):
            test_sampler = DistributedSampler(test_dataset, shuffle=True)
        else:
            test_sampler = None
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g
        )
        return train_loader, test_loader

    @staticmethod
    def get_model():
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion():
        """Get loss criterion for training."""
        # criterion is a class, must have compute_loss and compute_metrics
        return None

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": 5e-4, "lr": self.args.lr}
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer
    
    def clear_memory(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # 所有进程同步
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def main(self, collate_fn=default_collate):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(collate_fn)

        # Get model
        model = self.get_model()

        # Get criterion
        criterion = self.get_criterion()

        # Get optimizer
        optimizer = self.get_optimizer(model)

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        if (bool(self.args.distributed)):
            model = DistributedDataParallel(
                model, device_ids=[self.args.local_rank],
                broadcast_buffers=False, find_unused_parameters=True
            )

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint and os.path.isfile(self.args.checkpoint):
            # assert os.path.isfile(self.args.checkpoint)
            start_iter, best_loss = self.load_checkpoint(model, optimizer)

        # Eval only
        if bool(self.args.eval_only):
            print("Test evaluation.......")
            model.eval()
            new_loss = self.evaluate_nsteps(
                model, criterion, test_loader, step_id=-1,
                val_iters=max(
                    5,
                    int(2 * len(self.args.tasks)/self.args.batch_size_val)
                )
            )
            return model

        # Training loop
        iter_loader = iter(train_loader)
        model.train()
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            self.train_one_step(model, criterion, optimizer, step_id, sample)
            if (step_id + 1) % self.args.val_freq == 0:
                print("Train evaluation.......")
                model.eval()
                new_loss = self.evaluate_nsteps(
                    model, criterion, train_loader, step_id,
                    val_iters=max(
                        5,
                        int(4 * len(self.args.tasks)/self.args.batch_size_val)
                    ),
                    split='train'
                )
                self.clear_memory()
                print("Test evaluation.......")
                model.eval()
                new_loss = self.evaluate_nsteps(
                    model, criterion, test_loader, step_id,
                    val_iters=max(
                        5,
                        int(4 * len(self.args.tasks)/self.args.batch_size_val)
                    )
                )
                if (bool(self.args.distributed) and dist.get_rank() == 0) or not bool(self.args.distributed):  # save model
                    best_loss = self.save_checkpoint(
                        model, optimizer, step_id,
                        new_loss, best_loss
                    )
                model.train()
            
            # print(f"Epoch {step_id + 1}: Memory allocated = {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            # print(f"Epoch {step_id + 1}: Memory reserved = {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            
            self.clear_memory()

        return model

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        pass

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        return None

    def load_checkpoint(self, model, optimizer):
        """Load from checkpoint."""
        print("=> loading checkpoint '{}'".format(self.args.checkpoint))

        model_dict = torch.load(self.args.checkpoint, map_location="cpu")
        model.load_state_dict(model_dict["weight"], strict = False)
        if 'optimizer' in model_dict:
            optimizer.load_state_dict(model_dict["optimizer"])
            for p in range(len(optimizer.param_groups)):
                optimizer.param_groups[p]['lr'] = self.args.lr
        start_iter = model_dict.get("iter", 0)
        best_loss = model_dict.get("best_loss", None)

        print("=> loaded successfully '{}' (step {})".format(
            self.args.checkpoint, model_dict.get("iter", 0)
        ))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss

    def save_checkpoint(self, model, optimizer, step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        self.args.save_ckpt.mkdir(exist_ok=True, parents=True)
        print(f"Saving checkpoint in Path {self.args.save_ckpt}")
        if new_loss is None or best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            try:
                torch.save({
                    "weight": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": step_id + 1,
                    "best_loss": best_loss
                }, self.args.save_ckpt / f"best_{step_id}.pth")
            except:
                print("Failed to save best.")
        torch.save({
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.save_ckpt / "last.pth")
        return best_loss

    def synchronize_between_processes(self, a_dict):
        all_dicts = all_gather(a_dict)

        if not is_dist_avail_and_initialized() or dist.get_rank() == 0:
            merged = {}
            for key in all_dicts[0].keys():
                device = all_dicts[0][key].device
                merged[key] = torch.cat([
                    p[key].to(device) for p in all_dicts
                    if key in p
                ])
            a_dict = merged
        return a_dict


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)

    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    device = torch.device("cuda")
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    # storage = torch.UntypedStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty(
            (max_size,), dtype=torch.uint8, device=device
        ))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,),
            dtype=torch.uint8, device=device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def all_gather_cpu(data):
    """
    Proper CPU-only all_gather that handles device coordination correctly
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    # 1. Serialize on CPU
    buffer = pickle.dumps(data)
    length = torch.tensor([len(buffer)], dtype=torch.long, device="cuda")  # Small GPU tensor
    
    # 2. Gather sizes (using GPU for coordination)
    length_list = [torch.empty_like(length) for _ in range(world_size)]
    dist.all_gather(length_list, length)
    
    # 3. Prepare CPU buffers
    max_length = max(int(l.item()) for l in length_list)
    cpu_buffer = torch.empty(max_length, dtype=torch.uint8, pin_memory=True)
    cpu_buffer[:len(buffer)] = torch.frombuffer(buffer, dtype=torch.uint8)
    
    # 4. Gather on CPU using GPU-coordinated lengths
    gathered = [torch.empty_like(cpu_buffer) for _ in range(world_size)]
    dist.all_gather(gathered, cpu_buffer)
    
    # 5. Deserialize
    return [pickle.loads(g.cpu().numpy().tobytes()[:int(l.item())]) 
            for g, l in zip(gathered, length_list)]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
