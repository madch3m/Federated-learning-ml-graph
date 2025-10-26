import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from typing import Dict, Tuple
from copy import deepcopy
from tqdm import tqdm


class FederatedClient:
    """local client"""
    
    def __init__(
        self,
        client_id: int,
        dataset: Subset,
        local_epochs: int,
        batch_size: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        optimizer: str,
        device: str,):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.device = device
        self.model = None
        
    def get_dataset_size(self) -> int:
        return len(self.dataset)
    
    def train(self, global_model: nn.Module) -> Tuple[Dict[str, torch.Tensor], int]:
       
        # Create a local copy of the global model
        self.model = deepcopy(global_model).to(self.device)
        self.model.train()
        
        # Create data loader
        loader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,drop_last=False
        )
        
        # Setup loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if self.optimizer_name.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        
        # Local training loop
        for epoch in range(self.local_epochs):
            for x, y in tqdm(
                loader,
                desc=f"Client {self.client_id} | Epoch {epoch+1}/{self.local_epochs}",
                leave=False
            ):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
        
        # Return the trained model state and dataset size
        return deepcopy(self.model.state_dict()), len(self.dataset)