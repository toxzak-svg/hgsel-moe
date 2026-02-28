import torch
import torch.distributed as dist
import time
import numpy as np

# Microbenchmark for All-to-All Communication
# Measures raw all_to_all time vs local expert compute time

def simulate_routing_distribution(batch_size, num_experts, hidden_dim):
    """Simulate token routing distribution."""
    tokens_per_expert = batch_size // num_experts
    return {
        rank: torch.randn(tokens_per_expert, hidden_dim).cuda()
        for rank in range(num_experts)
    }

def measure_all_to_all_time(payloads, num_experts):
    """Measure all_to_all communication time."""
    start_time = time.time()
    output = [torch.empty_like(payloads[rank]) for rank in range(num_experts)]
    dist.all_to_all(output, list(payloads.values()))
    end_time = time.time()
    return end_time - start_time

def measure_local_compute_time(payloads):
    """Measure local expert compute time."""
    start_time = time.time()
    for rank, tokens in payloads.items():
        # Simulate local computation (e.g., forward pass)
        _ = tokens @ tokens.T
    end_time = time.time()
    return end_time - start_time

def main():
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Parameters
    batch_size = 1024
    hidden_dim = 512
    num_experts = world_size

    # Simulate token routing
    payloads = simulate_routing_distribution(batch_size, num_experts, hidden_dim)

    # Measure all_to_all time
    all_to_all_time = measure_all_to_all_time(payloads, num_experts)
    if rank == 0:
        print(f"All-to-All Time: {all_to_all_time:.6f} seconds")

    # Measure local compute time
    local_compute_time = measure_local_compute_time(payloads)
    if rank == 0:
        print(f"Local Compute Time: {local_compute_time:.6f} seconds")

if __name__ == "__main__":
    main()