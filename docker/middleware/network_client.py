import torch
import pickle
import requests
import time
import logging
from typing import Optional, Dict
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from client.local_server import FederatedClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkThrottleClient:
    
    def __init__(
        self,
        client: FederatedClient,
        server_url: str,
        client_id: str,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        request_timeout: int = 120,
        connection_timeout: int = 30,
    ):

        self.client = client
        self.server_url = server_url.rstrip('/')
        self.client_id = client_id
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.request_timeout = request_timeout
        self.connection_timeout = connection_timeout
        
        # Create session with retry logic
        self.session = self._create_session()
        
        # State tracking
        self.current_round = 0
        self.last_successful_update = None
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry configuration."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def wait_for_server(self, timeout: int = 300, poll_interval: int = 5) -> bool:

        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(
                    f"{self.server_url}/health",
                    timeout=self.connection_timeout
                )
                if response.status_code == 200:
                    logger.info(f"Client {self.client_id}: Server is available")
                    return True
            except requests.exceptions.RequestException as e:
                logger.debug(f"Client {self.client_id}: Server not ready yet - {e}")
            
            time.sleep(poll_interval)
        
        logger.error(f"Client {self.client_id}: Server did not become available within {timeout}s")
        return False
    
    def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Client {self.client_id}: Downloading global model (attempt {attempt + 1})")
                
                response = self.session.get(
                    f"{self.server_url}/model",
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                
                # Get round info from headers
                self.current_round = int(response.headers.get('X-Round', 0))
                
                # Deserialize model state
                model_state = pickle.loads(response.content)
                logger.info(f"Client {self.client_id}: Successfully downloaded model for round {self.current_round}")
                
                return model_state
                
            except requests.exceptions.Timeout:
                logger.warning(f"Client {self.client_id}: Download timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Client {self.client_id}: Download failed - {e} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
            except Exception as e:
                logger.error(f"Client {self.client_id}: Unexpected error during download - {e}")
                return None
        
        logger.error(f"Client {self.client_id}: Failed to download model after {self.max_retries} attempts")
        return None
    
    def upload_update(self, state_dict: Dict[str, torch.Tensor], num_samples: int) -> bool:
        """Upload model state to server."""
        # Serialize the model state
        serialized_state = pickle.dumps(state_dict)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Client {self.client_id}: Uploading update (attempt {attempt + 1})")
                
                # Send model state directly to submit_update endpoint
                response = self.session.post(
                    f"{self.server_url}/submit_update/{self.client_id}",
                    params={"num_samples": num_samples},
                    data=serialized_state,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Client {self.client_id}: Update submitted successfully - {result.get('updates_received', 0)}/{result.get('expected_clients', 0)} updates received")
                self.last_successful_update = time.time()
                
                return True
                
            except requests.exceptions.Timeout:
                logger.warning(f"Client {self.client_id}: Upload timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Client {self.client_id}: Upload failed - {e} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
            except Exception as e:
                logger.error(f"Client {self.client_id}: Unexpected error during upload - {e}")
                return False
        
        logger.error(f"Client {self.client_id}: Failed to upload update after {self.max_retries} attempts")
        return False
    
    def get_server_status(self) -> Optional[Dict]:
        
        try:
            response = self.session.get(
                f"{self.server_url}/status",
                timeout=self.connection_timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Client {self.client_id}: Failed to get status - {e}")
            return None
    
    def run_training_round(self, global_model_state: Dict[str, torch.Tensor]) -> bool:

        try:
            # Create a temporary model to load the global state
            # Use the same model type as the server
            from model import get_model
            import os
            
            # Get model name from environment or default to cnn
            model_name = os.getenv("MODEL_NAME", "cnn").lower().strip()
            global_model = get_model(model_name)
            global_model.load_state_dict(global_model_state)
            
            logger.info(f"Client {self.client_id}: Starting local training for round {self.current_round}")
            
            # Train locally
            state_dict, num_samples = self.client.train(global_model)
            
            logger.info(f"Client {self.client_id}: Training completed with {num_samples} samples")
            
            # Upload update
            success = self.upload_update(state_dict, num_samples)
            
            if success:
                logger.info(f"Client {self.client_id}: Round {self.current_round} completed successfully")
            else:
                logger.error(f"Client {self.client_id}: Failed to complete round {self.current_round}")
            
            return success
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Training round failed - {e}")
            return False
    
    def participate(self, num_rounds: int, start_delay: float = 0):

        if start_delay > 0:
            logger.info(f"Client {self.client_id}: Waiting {start_delay}s before starting")
            time.sleep(start_delay)
        
        # Wait for server to be ready
        if not self.wait_for_server():
            logger.error(f"Client {self.client_id}: Cannot connect to server, exiting")
            return
        
        for round_num in range(num_rounds):
            logger.info(f"Client {self.client_id}: Starting round {round_num + 1}/{num_rounds}")
            
            # Download global model
            global_model_state = self.get_global_model()
            if global_model_state is None:
                logger.error(f"Client {self.client_id}: Failed to download model, skipping round")
                continue
            
            # Run training round
            success = self.run_training_round(global_model_state)
            
            if not success:
                logger.warning(f"Client {self.client_id}: Round failed, will retry next round")
            
            # Wait for server to aggregate before next round
            time.sleep(5)  # Brief pause between rounds


def create_network_client(
    client_id: int,
    dataset,
    server_url: str,
    local_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 0.01,
    device: str = "cpu",
) -> NetworkThrottleClient:

    # Create base client
    base_client = FederatedClient(
        client_id=client_id,
        dataset=dataset,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=0.0,
        weight_decay=0.0,
        optimizer="sgd",
        device=device,
    )
    
    # Wrap with network tolerance
    network_client = NetworkThrottleClient(
        client=base_client,
        server_url=server_url,
        client_id=f"client_{client_id}",
        max_retries=5,
        backoff_factor=2.0,
    )
    
    return network_client