"""
Utility functions for initializing and managing distributed training.
"""
import os
import torch
import torch.distributed as dist
from typing import Optional


def init_distributed(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> bool:
    """
    Initialize distributed training environment.
    
    Args:
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU). Auto-detected if None.
        init_method: Initialization method (e.g., 'env://', 'tcp://localhost:23456')
        world_size: Number of processes. If None, read from environment.
        rank: Process rank. If None, read from environment.
    
    Returns:
        True if distributed was successfully initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    
    if dist.is_initialized():
        return True
    
    # Get values from environment if not provided
    if world_size is None:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if rank is None:
        rank = int(os.getenv("RANK", "-1"))
    
    if init_method is None:
        init_method = os.getenv("INIT_METHOD", "env://")
    
    # Auto-detect backend if not provided
    if backend is None:
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
    
    # Only initialize if we have multiple processes
    if world_size <= 1:
        return False
    
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to initialize distributed training: {e}")
        return False


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed_available() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the number of processes in the distributed group."""
    if is_distributed_available():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_distributed_available():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0

