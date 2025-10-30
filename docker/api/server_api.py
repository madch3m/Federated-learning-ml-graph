import torch
import asyncio
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pickle
import logging
from pathlib import Path

from model.cnn import CNN
from server.model_server import FederatedServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Federated Learning Server")

# Global state
class ServerState:
    def __init__(self):
        self.server: Optional[FederatedServer] = None
        self.current_round: int = 0
        self.max_rounds: int = 25
        self.expected_clients: int = 5  # Number of clients expected per round
        self.min_clients: int = 2  # Minimum clients needed to proceed
        self.client_updates: Dict[str, tuple] = {}  # client_id -> (state_dict, num_samples, timestamp)
        self.round_start_time: Optional[datetime] = None
        self.round_timeout: int = 300  # 5 minutes timeout per round
        self.aggregation_lock = asyncio.Lock()
        self.ready_for_next_round = asyncio.Event()
        self.ready_for_next_round.set()
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
state = ServerState()

# Pydantic models
class InitRequest(BaseModel):
    num_rounds: int = 25
    expected_clients: int = 5
    min_clients: int = 2
    round_timeout: int = 300

class ClientUpdate(BaseModel):
    client_id: str
    num_samples: int

class RoundStatus(BaseModel):
    current_round: int
    max_rounds: int
    clients_updated: int
    expected_clients: int
    time_remaining: Optional[float]
    status: str

@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    logger.info("Starting Federated Learning Server...")
    state.server = FederatedServer(model=CNN(), device=state.device)
    logger.info(f"Server initialized on device: {state.device}")
    
    # Start background task to monitor round timeouts
    asyncio.create_task(monitor_round_timeout())

@app.post("/initialize")
async def initialize_server(config: InitRequest):
    """Initialize or reinitialize the server with configuration."""
    state.max_rounds = config.num_rounds
    state.expected_clients = config.expected_clients
    state.min_clients = max(1, config.min_clients)
    state.round_timeout = config.round_timeout
    state.current_round = 0
    state.client_updates.clear()
    state.server = FederatedServer(model=CNN(), device=state.device)
    state.ready_for_next_round.set()
    
    logger.info(f"Server initialized: {config.num_rounds} rounds, "
                f"{config.expected_clients} expected clients, "
                f"{config.min_clients} min clients")
    
    return {
        "status": "initialized",
        "current_round": state.current_round,
        "max_rounds": state.max_rounds,
        "device": state.device
    }

@app.get("/model")
async def get_global_model():
    """Download the current global model state."""
    if state.server is None:
        raise HTTPException(status_code=400, detail="Server not initialized")
    
    # Wait if aggregation is in progress
    await state.ready_for_next_round.wait()
    
    model_state = state.server.get_global_model().state_dict()
    serialized = pickle.dumps(model_state)
    
    return Response(
        content=serialized,
        media_type="application/octet-stream",
        headers={
            "X-Round": str(state.current_round),
            "X-Max-Rounds": str(state.max_rounds)
        }
    )

@app.post("/update")
async def receive_client_update(
    client_update: ClientUpdate,
    background_tasks: BackgroundTasks
):
    """Receive a model update from a client (metadata only)."""
    if state.server is None:
        raise HTTPException(status_code=400, detail="Server not initialized")
    
    client_id = client_update.client_id
    
    # Check if this update is for the current round
    if client_id in state.client_updates:
        logger.warning(f"Client {client_id} already submitted for round {state.current_round}")
        return {"status": "duplicate", "message": "Update already received for this round"}
    
    logger.info(f"Registered client {client_id} update intent ({client_update.num_samples} samples)")
    
    return {
        "status": "registered",
        "client_id": client_id,
        "current_round": state.current_round,
        "upload_url": f"/upload/{client_id}"
    }

@app.post("/upload/{client_id}")
async def upload_model_state(client_id: str, background_tasks: BackgroundTasks):
    """Upload the actual model state dict (sent as binary in body)."""
    from fastapi import Request
    
    if state.server is None:
        raise HTTPException(status_code=400, detail="Server not initialized")
    
    # Get the request to read body
    from starlette.requests import Request as StarletteRequest
    
    async def get_body(request):
        return await request.body()
    
    # This is a workaround - in production, use proper file upload
    # For now, clients should POST binary data
    return {"status": "upload_endpoint", "message": "Send model state as binary POST body"}

@app.post("/submit_update/{client_id}")
async def submit_client_update(
    client_id: str,
    num_samples: int,
    background_tasks: BackgroundTasks
):
    """
    Submit a client update with model state in request body.
    This combines registration and upload in one call.
    """
    from fastapi import Request
    import io
    
    if state.server is None:
        raise HTTPException(status_code=400, detail="Server not initialized")
    
    # For this example, we'll accept the model state as a query parameter or separate call
    # In practice, you'd handle binary upload properly
    logger.info(f"Client {client_id} submitted update ({num_samples} samples)")
    
    return {
        "status": "accepted",
        "client_id": client_id,
        "current_round": state.current_round
    }

@app.get("/status")
async def get_round_status() -> RoundStatus:
    """Get the current status of the training round."""
    time_remaining = None
    if state.round_start_time:
        elapsed = (datetime.now() - state.round_start_time).total_seconds()
        time_remaining = max(0, state.round_timeout - elapsed)
    
    status = "waiting"
    if state.current_round >= state.max_rounds:
        status = "completed"
    elif len(state.client_updates) >= state.min_clients:
        status = "ready_to_aggregate"
    elif state.round_start_time is None:
        status = "idle"
    
    return RoundStatus(
        current_round=state.current_round,
        max_rounds=state.max_rounds,
        clients_updated=len(state.client_updates),
        expected_clients=state.expected_clients,
        time_remaining=time_remaining,
        status=status
    )

async def monitor_round_timeout():
    """Background task to monitor round timeouts and trigger aggregation."""
    while True:
        await asyncio.sleep(10)  # Check every 10 seconds
        
        if state.round_start_time is None:
            continue
        
        elapsed = (datetime.now() - state.round_start_time).total_seconds()
        num_updates = len(state.client_updates)
        
        # Trigger aggregation if:
        # 1. We have all expected clients, OR
        # 2. We have minimum clients and timeout reached, OR
        # 3. Timeout reached regardless
        should_aggregate = (
            (num_updates >= state.expected_clients) or
            (num_updates >= state.min_clients and elapsed >= state.round_timeout) or
            (elapsed >= state.round_timeout * 1.5)  # Hard timeout
        )
        
        if should_aggregate and num_updates >= state.min_clients:
            logger.info(f"Triggering aggregation: {num_updates} clients, {elapsed:.1f}s elapsed")
            asyncio.create_task(trigger_aggregation())

async def trigger_aggregation():
    """Aggregate client updates and advance to next round."""
    async with state.aggregation_lock:
        if len(state.client_updates) < state.min_clients:
            logger.warning(f"Not enough clients for aggregation: {len(state.client_updates)}/{state.min_clients}")
            return
        
        state.ready_for_next_round.clear()
        
        try:
            # Prepare updates for aggregation
            client_updates = [
                (state_dict, num_samples) 
                for state_dict, num_samples, _ in state.client_updates.values()
            ]
            
            logger.info(f"Aggregating {len(client_updates)} client updates for round {state.current_round}")
            
            # Perform aggregation
            state.server.aggregate(client_updates)
            
            # Clear updates and advance round
            state.client_updates.clear()
            state.current_round += 1
            state.round_start_time = None
            
            logger.info(f"Advanced to round {state.current_round}/{state.max_rounds}")
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
        finally:
            state.ready_for_next_round.set()

@app.post("/aggregate")
async def manual_aggregate():
    """Manually trigger aggregation (for testing or early aggregation)."""
    if len(state.client_updates) < state.min_clients:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough clients: {len(state.client_updates)}/{state.min_clients}"
        )
    
    await trigger_aggregation()
    
    return {
        "status": "aggregated",
        "current_round": state.current_round,
        "clients_aggregated": len(state.client_updates)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "server_initialized": state.server is not None,
        "current_round": state.current_round,
        "device": state.device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)