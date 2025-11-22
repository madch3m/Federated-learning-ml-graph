import torch
import asyncio
import uuid
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pickle
import logging
from pathlib import Path
from torch.utils.data import Dataset

from model import get_model, get_model_info
from server.model_server import FederatedServer
from torchvision import datasets, transforms
from utils.plotter import plot_history
from utils.data_loader import get_dataset_for_model

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
        self.expected_clients: int = 5  
        self.min_clients: int = 2  
        self.client_updates: Dict[str, tuple] = {}  
        self.round_start_time: Optional[datetime] = None
        self.round_timeout: int = 300  
        self.aggregation_lock = asyncio.Lock()
        self.ready_for_next_round = asyncio.Event()
        self.ready_for_next_round.set()
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # Evaluation tracking
        self.evaluation_history: Dict[str, List[float]] = {"round": [], "acc": [], "loss": []}
        self.testset: Optional[Dataset] = None
        self.experiments_dir: Path = Path("/app/experiments")
        self.last_experiment_dir: Optional[Path] = None
        self.model_name: str = "cnn"  # Default model
        
state = ServerState()

# Pydantic models
class InitRequest(BaseModel):
    num_rounds: int = 25
    expected_clients: int = 5
    min_clients: int = 2
    round_timeout: int = 300
    model_name: str = "cnn"  # Model to use: "cnn", "cifar10_cnn", or "resnet"

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
    
    # Get model name from environment variable or use default
    model_name = os.getenv("MODEL_NAME", "cnn").lower().strip()
    state.model_name = model_name
    
    # Load the selected model
    try:
        model = get_model(model_name)
        state.server = FederatedServer(model=model, device=state.device)
        model_info = get_model_info(model_name)
        logger.info(f"Server initialized on device: {state.device}")
        logger.info(f"Model: {model_info.get('name', model_name)} - {model_info.get('description', '')}")
        if state.server.use_distributed:
            logger.info(f"Distributed aggregation enabled: {state.server.world_size} processes")
        else:
            logger.info("Using optimized single-device aggregation")
    except ValueError as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        logger.info("Falling back to default CNN model")
        model = get_model("cnn")
        state.server = FederatedServer(model=model, device=state.device)
        state.model_name = "cnn"
    
    # Load appropriate test dataset based on model
    dataset_name = "CIFAR10" if "cifar10" in model_name else "MNIST"
    logger.info(f"Loading {dataset_name} test dataset for model '{model_name}'...")
    _, state.testset = get_dataset_for_model(model_name, data_dir="./data")
    logger.info(f"{dataset_name} test dataset loaded: {len(state.testset)} samples")
    
    # Evaluate initial model (round 0)
    if state.testset is not None:
        acc0, loss0 = state.server.evaluate(state.testset)
        state.evaluation_history["round"].append(0)
        state.evaluation_history["acc"].append(acc0)
        state.evaluation_history["loss"].append(loss0)
        logger.info(f"[Round 00] Initial Test Acc: {acc0*100:5.2f}% | Test Loss: {loss0:.4f}")
    
    # Ensure experiments directory exists
    state.experiments_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiments directory: {state.experiments_dir}")
    
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
    
    # Load the specified model
    model_name = config.model_name.lower().strip() if hasattr(config, 'model_name') else state.model_name
    try:
        model = get_model(model_name)
        state.server = FederatedServer(model=model, device=state.device)
        state.model_name = model_name
        model_info = get_model_info(model_name)
        logger.info(f"Model loaded: {model_info.get('name', model_name)}")
    except ValueError as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        logger.info("Falling back to default CNN model")
        model = get_model("cnn")
        state.server = FederatedServer(model=model, device=state.device)
        state.model_name = "cnn"
    
    state.ready_for_next_round.set()
    
    # Reset evaluation history and evaluate initial model
    state.evaluation_history = {"round": [], "acc": [], "loss": []}
    if state.testset is not None:
        acc0, loss0 = state.server.evaluate(state.testset)
        state.evaluation_history["round"].append(0)
        state.evaluation_history["acc"].append(acc0)
        state.evaluation_history["loss"].append(loss0)
        logger.info(f"[Round 00] Initial Test Acc: {acc0*100:5.2f}% | Test Loss: {loss0:.4f}")
    
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
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Submit a client update with model state in request body (pickled binary).
    This combines registration and upload in one call.
    """
    if state.server is None:
        raise HTTPException(status_code=400, detail="Server not initialized")
    
    # Check if this update is for the current round
    if client_id in state.client_updates:
        logger.warning(f"Client {client_id} already submitted for round {state.current_round}")
        return {"status": "duplicate", "message": "Update already received for this round"}
    
    # Read the binary model state from request body
    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty request body - model state required")
        
        # Deserialize the model state
        state_dict = pickle.loads(body)
        
        # Store the update
        state.client_updates[client_id] = (state_dict, num_samples, datetime.now())
        
        # Set round start time if this is the first update
        if state.round_start_time is None:
            state.round_start_time = datetime.now()
        
        logger.info(f"Client {client_id} submitted update ({num_samples} samples) for round {state.current_round}")
        logger.info(f"Total updates received: {len(state.client_updates)}/{state.expected_clients}")
        
        # Check if we have enough clients to aggregate immediately
        if len(state.client_updates) >= state.expected_clients:
            logger.info(f"Received all expected updates, triggering aggregation...")
            asyncio.create_task(trigger_aggregation())
        
        return {
            "status": "accepted",
            "client_id": client_id,
            "current_round": state.current_round,
            "updates_received": len(state.client_updates),
            "expected_clients": state.expected_clients
        }
        
    except pickle.UnpicklingError as e:
        logger.error(f"Failed to unpickle model state from client {client_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid model state format: {e}")
    except Exception as e:
        logger.error(f"Error processing update from client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing update: {e}")

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
            
            # Evaluate model after aggregation
            if state.testset is not None and state.server is not None:
                acc, loss = state.server.evaluate(state.testset)
                state.evaluation_history["round"].append(state.current_round)
                state.evaluation_history["acc"].append(acc)
                state.evaluation_history["loss"].append(loss)
                logger.info(f"[Round {state.current_round:02d}] Test Acc: {acc*100:5.2f}% | Test Loss: {loss:.4f}")
            
            # Check if training is complete
            if state.current_round >= state.max_rounds:
                logger.info("="*60)
                logger.info("Training completed! Saving model and generating visualization...")
                logger.info("="*60)
                await handle_training_completion()
            
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

async def handle_training_completion():
    """Handle post-training tasks: save model, evaluate, and generate visualization."""
    try:
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = state.experiments_dir / f"federated_run_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        state.last_experiment_dir = exp_dir  # Store for API access
        
        logger.info(f"Experiment directory created: {exp_dir}")
        
        # Save the model
        model_path = exp_dir / "mnist_cnn.pt"
        if state.server is not None:
            state.server.save_model(str(model_path))
            logger.info(f"Model saved to: {model_path}")
        
        # Run final evaluation on saved model
        if state.testset is not None and state.server is not None:
            logger.info("Running final evaluation on saved model...")
            final_acc, final_loss = state.server.evaluate(state.testset)
            
            logger.info("="*60)
            logger.info("FINAL MODEL EVALUATION RESULTS")
            logger.info("="*60)
            logger.info(f"Final Test Accuracy: {final_acc*100:5.2f}%")
            logger.info(f"Final Test Loss: {final_loss:.4f}")
            logger.info(f"Total Rounds: {len(state.evaluation_history['round'])}")
            
            # Display summary of all rounds
            if state.evaluation_history["round"]:
                logger.info("\nTraining Summary:")
                logger.info("-" * 60)
                for rnd, acc, loss in zip(
                    state.evaluation_history["round"],
                    state.evaluation_history["acc"],
                    state.evaluation_history["loss"]
                ):
                    logger.info(f"Round {rnd:2d}: Acc={acc*100:5.2f}% | Loss={loss:.4f}")
            
            logger.info("="*60)
            
            # Generate and save visualization
            logger.info("Generating visualization...")
            try:
                plot_history(
                    state.evaluation_history,
                    exp_dir=exp_dir,
                    show=False  # Don't show in Docker, just save
                )
                logger.info(f"Visualization saved to: {exp_dir / 'metrics.png'}")
                logger.info(f"Metrics CSV saved to: {exp_dir / 'metrics.csv'}")
            except Exception as e:
                logger.error(f"Failed to generate visualization: {e}")
            
            logger.info("="*60)
            logger.info(f"Experiment completed successfully!")
            logger.info(f"All outputs saved to: {exp_dir}")
            logger.info("="*60)
        else:
            logger.warning("Cannot evaluate: testset or server not available")
            
    except Exception as e:
        logger.error(f"Error during training completion: {e}", exc_info=True)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "server_initialized": state.server is not None,
        "current_round": state.current_round,
        "device": state.device
    }

@app.get("/results")
async def get_results():
    """Get experiment results and evaluation history."""
    if state.server is None:
        raise HTTPException(status_code=400, detail="Server not initialized")
    
    return {
        "current_round": state.current_round,
        "max_rounds": state.max_rounds,
        "evaluation_history": state.evaluation_history,
        "last_experiment_dir": str(state.last_experiment_dir) if state.last_experiment_dir else None,
        "training_complete": state.current_round >= state.max_rounds,
        "summary": {
            "total_rounds": len(state.evaluation_history.get("round", [])),
            "final_accuracy": state.evaluation_history["acc"][-1] * 100 if state.evaluation_history["acc"] else None,
            "final_loss": state.evaluation_history["loss"][-1] if state.evaluation_history["loss"] else None,
            "best_accuracy": max(state.evaluation_history["acc"]) * 100 if state.evaluation_history["acc"] else None,
            "best_round": state.evaluation_history["round"][state.evaluation_history["acc"].index(max(state.evaluation_history["acc"]))] if state.evaluation_history["acc"] else None,
        }
    }

@app.get("/results/metrics")
async def get_metrics_csv():
    """Get metrics CSV content."""
    if state.last_experiment_dir is None:
        raise HTTPException(status_code=404, detail="No experiment results available yet")
    
    csv_path = state.last_experiment_dir / "metrics.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Metrics CSV not found")
    
    import csv as csv_module
    from fastapi.responses import Response
    
    with csv_path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    return Response(content=content, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=metrics.csv"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)