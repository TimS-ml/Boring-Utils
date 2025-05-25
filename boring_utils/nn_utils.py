'''
some of them grabed from x_transformer
'''

import math
import numpy as np
import os
import re
import time
from functools import partial, wraps
from inspect import isfunction
from typing import (
    List, Dict, Tuple, Callable, Optional, Union, Any, 
    Iterator, TypeVar, Generic, cast, Iterable, Generator
)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


'''
For Training
'''
def resume_checkpoint(checkpoint_path: Optional[str], name_pattern: str = r".*[_-](\d+)\.(pt|pth|ckpt|bin|model|safetensors)$") -> Optional[str]:
    """
    If checkpoint_path is a directory, it will identify the latest checkpoint file in the directory. Name pattern needed. By default, it assumes the training step is the last number in the filename.
    """
    if checkpoint_path is None:
        return None
        
    if os.path.isdir(checkpoint_path):
        checkpoint_files = []
        for f_name in os.listdir(checkpoint_path):
            match = re.fullmatch(name_pattern, f_name)
            if match:
                step = int(match.group(1))
                checkpoint_files.append((step, os.path.join(checkpoint_path, f_name)))
        
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: x[0], reverse=True)  # Sort by step, descending
            resolved_checkpoint_file = checkpoint_files[0][1]
            print(f"Identified latest checkpoint in directory '{checkpoint_path}': '{resolved_checkpoint_file}'")
        else:
            print(f"Warning: No checkpoint files found in directory '{checkpoint_path}'.")
            return None
    elif os.path.isfile(checkpoint_path):
        resolved_checkpoint_file = checkpoint_path
    else:
        print(f"Warning: checkpoint_path path '{checkpoint_path}' is not a valid file or directory.")
        return None

    return resolved_checkpoint_file


def calculate_optimizer_stats(optimizer: torch.optim.Optimizer, prefix: str = 'optim') -> Dict[str, float]:
    """
    Calculate optimizer statistics and return them as a dictionary
    
    Args:
        optimizer: PyTorch optimizer
        prefix: Prefix for the metrics keys
    
    Returns:
        Dict[str, float]: Dictionary of optimizer statistics
    """
    metrics = {}
    
    # Calculate statistics for each parameter group
    for i, param_group in enumerate(optimizer.param_groups):
        # Current learning rate
        metrics[f"{prefix}/lr_group_{i}"] = param_group['lr']
        
        # Process each parameter
        for j, p in enumerate(param_group['params']):
            if not p.requires_grad or p.grad is None:
                continue
                
            state = optimizer.state[p]
            
            # For Adam/AdamW optimizers
            if 'exp_avg' in state:
                metrics[f"{prefix}/exp_avg_norm_group_{i}"] = state['exp_avg'].norm().item()
            
            if 'exp_avg_sq' in state:
                metrics[f"{prefix}/exp_avg_sq_norm_group_{i}"] = state['exp_avg_sq'].norm().item()
            
            # Gradient norm
            metrics[f"{prefix}/grad_norm_group_{i}"] = p.grad.norm().item()
            
            # Parameter norm
            metrics[f"{prefix}/param_norm_group_{i}"] = p.data.norm().item()
            
            # Only process a few parameters to avoid excessive data
            if j >= 3:
                break
    
    return metrics


def calculate_throughput(
        batch_size: int, 
        seq_len: int, 
        start_time: float, 
        end_time: float, 
        grad_accum_steps: int = 1,
        num_devices: int = 1) -> Dict[str, float]:
    """
    Calculate training throughput metrics
    
    Args:
        batch_size: Batch size per device
        seq_len: Sequence length
        start_time: Start time of the operation
        end_time: End time of the operation
        grad_accum_steps: Gradient accumulation steps
        num_devices: Number of devices used for training
    
    Returns:
        Dict[str, float]: Dictionary of throughput metrics
    """
    elapsed_time = end_time - start_time
    
    # Calculate overall throughput
    total_batch_size = batch_size * num_devices
    tokens_per_sec = total_batch_size * seq_len * grad_accum_steps / elapsed_time
    
    # Calculate per-device throughput
    device_throughput = tokens_per_sec / num_devices
    
    return {
        "throughput/tokens_per_sec": tokens_per_sec,
        "throughput/tokens_per_sec_per_device": device_throughput,
        "throughput/batch_time_sec": elapsed_time
    }


'''
From nano-GPT 
'''
def get_batch_np(
        data: np.ndarray, 
        block_size: int = 1024, 
        batch_size: int = 32, 
        device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive)
    # The shape of the tensor is defined by the variable argument size
    # 0 ~ len(data) - block_size with output shape of (batch_size,)
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([
        torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
        for i in ix
    ])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss_np(
        model: nn.Module,
        eval_iters: int,
        train_data: np.ndarray,
        val_data: np.ndarray,
        block_size: int = 1024,
        batch_size: int = 32,
        device: str = 'cpu') -> Dict[str, float]:
    out = {}
    model.eval()
    data_dic = {'train': train_data, 'val': val_data}
    for split, data in data_dic.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_np(data,
                                block_size=block_size,
                                batch_size=batch_size,
                                device=device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(
        it: int, 
        learning_rate: float, 
        warmup_iters: int, 
        lr_decay_iters: int, 
        min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)



'''
From X-transformer
'''
T = TypeVar('T')

def cycle(loader: Iterable[T]) -> Generator[T, None, None]:
    while True:
        for data in loader:
            yield data

class always:
    def __init__(self, val: Any) -> None:
        self.val = val

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.val


def exists(val: Optional[Any]) -> bool:
    return val is not None


def default(val: Optional[T], d: Union[T, Callable[[], T]]) -> T:
    if exists(val):
        return cast(T, val)
    return d() if isfunction(d) else cast(T, d)


def max_neg_value(tensor: torch.Tensor) -> float:
    return -torch.finfo(tensor.dtype).max


def pad_at_dim(
        t: torch.Tensor, 
        pad: Tuple[int, int], 
        dim: int = -1, 
        value: float = 0.) -> torch.Tensor:
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


# init helpers
def init_zero_(layer: nn.Module) -> None:
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)


# keyword argument helpers
def pick_and_pop(keys: List[str], d: Dict[str, Any]) -> Dict[str, Any]:
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(
        cond: Callable[[str], bool], 
        d: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return cast(Tuple[Dict[str, Any], Dict[str, Any]], tuple(return_val))


def string_begins_with(prefix: str, s: str) -> bool:
    return s.startswith(prefix)


def group_by_key_prefix(
        prefix: str, 
        d: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(
        prefix: str, 
        d: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix):], x[1]),
            tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs