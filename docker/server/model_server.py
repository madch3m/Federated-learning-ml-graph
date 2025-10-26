import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from copy import deepcopy


class FederatedServer:
    """Represents the central server in federated learning."""
    
    def __init__(self, model: nn.Module, device: str):
        self.global_model = model.to(device)
        self.device = device
        
    def get_global_model(self) -> nn.Module:
        """Return the current global model."""
        return self.global_model
    
    @torch.no_grad()
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> None:
        """
        Aggregate client model updates using FedAvg (weighted averaging).
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        # Calculate total number of samples across all clients
        total_samples = sum(n_samples for _, n_samples in client_updates)
        
        # Initialize the aggregated state dictionary
        avg_state: Dict[str, torch.Tensor] = {}
        
        for key, value in self.global_model.state_dict().items():
            if value.dtype.is_floating_point or value.dtype.is_complex:
                # Initialize with zeros for parameters that will be averaged
                avg_state[key] = torch.zeros_like(value, device=self.device, dtype=value.dtype)
            else:
                # For non-floating point tensors (e.g., batch norm running stats), just clone
                avg_state[key] = value.clone().to(self.device)
        
        # Perform weighted averaging
        for state_dict, n_samples in client_updates:
            weight = n_samples / total_samples
            for key in avg_state.keys():
                tensor = state_dict[key].to(self.device)
                if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
                    avg_state[key] += tensor * weight
                else:
                    # For non-floating point, use the last client's value
                    avg_state[key] = tensor
        
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