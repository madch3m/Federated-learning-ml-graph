import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import os

try:
    from utils.distributed_utils import is_distributed_available
except ImportError:
    # Fallback if utils module not available
    def is_distributed_available():
        return dist.is_available() and dist.is_initialized()


class FederatedServer:
    """Represents the central server in federated learning."""
    
    def __init__(self, model: nn.Module, device: str, use_distributed: Optional[bool] = None):
        self.global_model = model.to(device)
        self.device = device
        
        # Auto-detect distributed mode if not explicitly set
        if use_distributed is None:
            use_distributed = os.getenv("USE_DISTRIBUTED", "false").lower() in ("true", "1", "yes")
        
        self.use_distributed = use_distributed and is_distributed_available()
        
        if self.use_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            # Use NCCL backend for GPU, Gloo for CPU
            if device.startswith("cuda"):
                self.backend = "nccl"
            else:
                self.backend = "gloo"
        else:
            self.world_size = 1
            self.rank = 0
        
    def get_global_model(self) -> nn.Module:
        """Return the current global model."""
        return self.global_model
    
    @torch.no_grad()
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> None:
        """
        Aggregate client model updates using FedAvg (weighted averaging).
        Uses torch.distributed for parallel aggregation when available.
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Calculate total number of samples across all clients
        total_samples = sum(n_samples for _, n_samples in client_updates)
        
        # Get model state dict keys for consistent ordering
        model_keys = list(self.global_model.state_dict().keys())
        
        if self.use_distributed and self.world_size > 1:
            # Distributed aggregation path - parallelize across GPUs
            self._aggregate_distributed(client_updates, total_samples, model_keys)
        else:
            # Optimized single-device aggregation
            self._aggregate_optimized(client_updates, total_samples, model_keys)
    
    @torch.no_grad()
    def _aggregate_optimized(self, client_updates: List[Tuple[Dict[str, torch.Tensor], int]], 
                            total_samples: int, model_keys: List[str]) -> None:
        """
        Optimized aggregation using vectorized operations and batch processing.
        """
        # Pre-compute weights
        weights = [n_samples / total_samples for _, n_samples in client_updates]
        
        # Initialize aggregated state dictionary
        avg_state: Dict[str, torch.Tensor] = {}
        global_state = self.global_model.state_dict()
        
        # Process each parameter tensor
        for key in model_keys:
            global_tensor = global_state[key]
            
            if global_tensor.dtype.is_floating_point or global_tensor.dtype.is_complex:
                # Initialize with zeros for weighted averaging
                avg_state[key] = torch.zeros_like(global_tensor, device=self.device, dtype=global_tensor.dtype)
                
                # Vectorized weighted sum: accumulate all client updates at once
                for (state_dict, _), weight in zip(client_updates, weights):
                    client_tensor = state_dict[key].to(self.device, non_blocking=True)
                    avg_state[key].add_(client_tensor, alpha=weight)
            else:
                # For non-floating point tensors (e.g., batch norm running stats)
                # Use the last client's value or majority vote
                if client_updates:
                    last_tensor = client_updates[-1][0][key].to(self.device, non_blocking=True)
                    avg_state[key] = last_tensor.clone()
                else:
                    avg_state[key] = global_tensor.clone().to(self.device)
        
        # Update the global model with aggregated parameters
        self.global_model.load_state_dict(avg_state)
    
    @torch.no_grad()
    def _aggregate_distributed(self, client_updates: List[Tuple[Dict[str, torch.Tensor], int]], 
                              total_samples: int, model_keys: List[str]) -> None:
        """
        Distributed aggregation using torch.distributed for multi-GPU parallelization.
        Splits client updates across available GPUs for parallel processing.
        """
        # Split client updates across available processes
        updates_per_process = len(client_updates) // self.world_size
        start_idx = self.rank * updates_per_process
        end_idx = start_idx + updates_per_process if self.rank < self.world_size - 1 else len(client_updates)
        local_updates = client_updates[start_idx:end_idx]
        
        # Calculate local total samples
        local_total_samples = sum(n_samples for _, n_samples in local_updates)
        
        # All-reduce to get global total samples across all processes
        local_total_tensor = torch.tensor(float(local_total_samples), device=self.device)
        dist.all_reduce(local_total_tensor, op=dist.ReduceOp.SUM)
        global_total_samples = local_total_tensor.item()
        
        # Initialize aggregated state dictionary
        avg_state: Dict[str, torch.Tensor] = {}
        global_state = self.global_model.state_dict()
        
        # Process each parameter tensor with distributed aggregation
        for key in model_keys:
            global_tensor = global_state[key]
            
            if global_tensor.dtype.is_floating_point or global_tensor.dtype.is_complex:
                # Initialize local aggregation
                local_avg = torch.zeros_like(global_tensor, device=self.device, dtype=global_tensor.dtype)
                
                # Aggregate local client updates
                for state_dict, n_samples in local_updates:
                    weight = n_samples / global_total_samples
                    client_tensor = state_dict[key].to(self.device, non_blocking=True)
                    local_avg.add_(client_tensor, alpha=weight)
                
                # All-reduce across all processes to get global average
                dist.all_reduce(local_avg, op=dist.ReduceOp.SUM)
                avg_state[key] = local_avg
            else:
                # For non-floating point tensors, use the last client's value
                if local_updates:
                    last_tensor = local_updates[-1][0][key].to(self.device, non_blocking=True)
                    # Broadcast from rank 0 to ensure consistency
                    if self.rank == 0:
                        avg_state[key] = last_tensor.clone()
                    else:
                        avg_state[key] = torch.zeros_like(global_tensor, device=self.device, dtype=global_tensor.dtype)
                    dist.broadcast(avg_state[key], src=0)
                else:
                    avg_state[key] = global_tensor.clone().to(self.device)
        
        # Update the global model with aggregated parameters
        self.global_model.load_state_dict(avg_state)
    
    @torch.no_grad()
    def evaluate(self, testset: Dataset, batch_size: int = 512) -> Tuple[float, float]:
    
        self.global_model.eval()
        loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.global_model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / total
        
        return accuracy, avg_loss
    
    def save_model(self, path: str) -> None:
        """Save the global model state dict to a file."""
        torch.save(self.global_model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """Load a model state dict from a file."""
        self.global_model.load_state_dict(torch.load(path, map_location=self.device))