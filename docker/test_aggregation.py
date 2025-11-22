#!/usr/bin/env python3
"""
Test script for the distributed aggregation implementation in FederatedServer.
Tests both optimized single-device and distributed aggregation modes.
"""
import torch
import torch.nn as nn
import time
import sys
from typing import Dict, List, Tuple

# Add current directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.model_server import FederatedServer
from model import get_model


def create_test_model(model_name: str = "cnn") -> nn.Module:
    """Create a test model."""
    return get_model(model_name)


def create_client_updates(model: nn.Module, num_clients: int = 5, device: str = "cpu") -> List[Tuple[Dict[str, torch.Tensor], int]]:
    """
    Create synthetic client updates for testing.
    Each client has slightly different weights and different sample counts.
    """
    client_updates = []
    base_state = model.state_dict()
    
    for i in range(num_clients):
        # Create a modified state dict for this client
        client_state = {}
        for key, value in base_state.items():
            if value.dtype.is_floating_point:
                # Add small random perturbations
                noise = torch.randn_like(value) * 0.1
                client_state[key] = (value + noise).to(device)
            else:
                client_state[key] = value.clone().to(device)
        
        # Vary sample counts (simulating different client data sizes)
        num_samples = 100 + i * 50
        
        client_updates.append((client_state, num_samples))
    
    return client_updates


def test_aggregation_correctness(model_name: str = "cnn", device: str = "cpu"):
    """Test that aggregation produces correct weighted averages."""
    print(f"\n{'='*60}")
    print(f"Testing Aggregation Correctness - Model: {model_name}, Device: {device}")
    print(f"{'='*60}")
    
    # Create model and server
    model = create_test_model(model_name)
    server = FederatedServer(model=model, device=device, use_distributed=False)
    
    # Get initial model state
    initial_state = server.get_global_model().state_dict()
    
    # Create client updates
    num_clients = 5
    client_updates = create_client_updates(model, num_clients, device)
    
    # Calculate expected total samples
    total_samples = sum(n_samples for _, n_samples in client_updates)
    print(f"Total clients: {num_clients}")
    print(f"Total samples: {total_samples}")
    
    # Perform aggregation
    start_time = time.time()
    server.aggregate(client_updates)
    aggregation_time = time.time() - start_time
    
    # Get aggregated state
    aggregated_state = server.get_global_model().state_dict()
    
    # Verify aggregation correctness for a sample parameter
    sample_key = list(initial_state.keys())[0]
    initial_param = initial_state[sample_key]
    aggregated_param = aggregated_state[sample_key]
    
    # Manually compute expected weighted average
    expected_param = torch.zeros_like(initial_param)
    for state_dict, n_samples in client_updates:
        weight = n_samples / total_samples
        expected_param += state_dict[sample_key] * weight
    
    # Check if aggregated matches expected (within numerical precision)
    diff = torch.abs(aggregated_param - expected_param).max().item()
    tolerance = 1e-5
    
    print(f"\nAggregation Time: {aggregation_time*1000:.2f} ms")
    print(f"Sample Parameter: {sample_key}")
    print(f"Max difference from expected: {diff:.2e}")
    print(f"Tolerance: {tolerance:.2e}")
    
    if diff < tolerance:
        print("✓ Aggregation correctness test PASSED")
        return True
    else:
        print("✗ Aggregation correctness test FAILED")
        return False


def test_aggregation_performance(model_name: str = "cnn", device: str = "cpu", num_clients: int = 10):
    """Test aggregation performance with varying numbers of clients."""
    print(f"\n{'='*60}")
    print(f"Testing Aggregation Performance - Model: {model_name}, Device: {device}")
    print(f"{'='*60}")
    
    model = create_test_model(model_name)
    server = FederatedServer(model=model, device=device, use_distributed=False)
    
    # Test with different numbers of clients
    client_counts = [5, 10, 20, 50]
    results = []
    
    for num_clients in client_counts:
        client_updates = create_client_updates(model, num_clients, device)
        
        # Warm-up
        server.aggregate(client_updates)
        
        # Time multiple aggregations
        num_runs = 5
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            server.aggregate(client_updates)
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        results.append((num_clients, avg_time))
        print(f"  {num_clients:2d} clients: {avg_time*1000:6.2f} ms (avg over {num_runs} runs)")
    
    print("\nPerformance Summary:")
    for num_clients, avg_time in results:
        print(f"  {num_clients:2d} clients -> {avg_time*1000:6.2f} ms")
    
    return results


def test_distributed_detection():
    """Test that distributed mode is correctly detected."""
    print(f"\n{'='*60}")
    print("Testing Distributed Mode Detection")
    print(f"{'='*60}")
    
    model = create_test_model("cnn")
    
    # Test without distributed
    server1 = FederatedServer(model=model, device="cpu", use_distributed=False)
    print(f"  use_distributed=False: {server1.use_distributed}")
    assert not server1.use_distributed, "Should not use distributed when explicitly disabled"
    
    # Test with distributed (should auto-detect from env)
    import os
    original_env = os.getenv("USE_DISTRIBUTED")
    os.environ["USE_DISTRIBUTED"] = "false"
    
    server2 = FederatedServer(model=model, device="cpu", use_distributed=None)
    print(f"  use_distributed=None (env=false): {server2.use_distributed}")
    assert not server2.use_distributed, "Should not use distributed when env is false"
    
    if original_env:
        os.environ["USE_DISTRIBUTED"] = original_env
    else:
        os.environ.pop("USE_DISTRIBUTED", None)
    
    print("✓ Distributed detection test PASSED")
    return True


def test_different_models():
    """Test aggregation with different model architectures."""
    print(f"\n{'='*60}")
    print("Testing Aggregation with Different Models")
    print(f"{'='*60}")
    
    models_to_test = ["cnn", "cifar10_cnn", "resnet"]
    results = []
    
    for model_name in models_to_test:
        try:
            print(f"\n  Testing {model_name}...")
            model = create_test_model(model_name)
            server = FederatedServer(model=model, device="cpu", use_distributed=False)
            
            client_updates = create_client_updates(model, num_clients=5, device="cpu")
            
            start_time = time.time()
            server.aggregate(client_updates)
            aggregation_time = time.time() - start_time
            
            results.append((model_name, aggregation_time))
            print(f"    ✓ {model_name}: {aggregation_time*1000:.2f} ms")
        except Exception as e:
            print(f"    ✗ {model_name}: Failed - {e}")
            results.append((model_name, None))
    
    return results


def main():
    """Run all tests."""
    print("="*60)
    print("FederatedServer Aggregation Test Suite")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    test_results = []
    
    # Test 1: Aggregation correctness
    try:
        result = test_aggregation_correctness("cnn", device)
        test_results.append(("Correctness", result))
    except Exception as e:
        print(f"✗ Correctness test failed: {e}")
        test_results.append(("Correctness", False))
    
    # Test 2: Performance
    try:
        test_aggregation_performance("cnn", device, num_clients=10)
        test_results.append(("Performance", True))
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        test_results.append(("Performance", False))
    
    # Test 3: Distributed detection
    try:
        result = test_distributed_detection()
        test_results.append(("Distributed Detection", result))
    except Exception as e:
        print(f"✗ Distributed detection test failed: {e}")
        test_results.append(("Distributed Detection", False))
    
    # Test 4: Different models
    try:
        test_different_models()
        test_results.append(("Different Models", True))
    except Exception as e:
        print(f"✗ Different models test failed: {e}")
        test_results.append(("Different Models", False))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:25s}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

