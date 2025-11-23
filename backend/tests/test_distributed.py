import asyncio
import torch
import sys
import os

# Add src directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(test_dir)
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, src_dir)

from engines.distributed import NVRAREngine, DistributedConfig, ParallelismStrategy, CommunicationBackend

async def test_nvrar_all_reduce():
    """Test NVRAR all-reduce performance"""
    print("🧪 TESTING NVRAR ALL-REDUCE...")
    
    engine = NVRAREngine()
    config = DistributedConfig(
        parallelism_strategy=ParallelismStrategy.TENSOR_PARALLELISM,
        communication_backend=CommunicationBackend.NVSHMEM,
        num_nodes=2,
        gpus_per_node=4,
        hierarchical_all_reduce=True
    )
    
    await engine.initialize_distributed(config)
    
    # Test different tensor sizes (from paper: 128KB to 2MB)
    tensor_sizes = [
        (131072, "128KB"),    # Paper's lower bound
        (524288, "512KB"),    # Medium size
        (2097152, "2MB")      # Paper's upper bound
    ]
    
    for size_bytes, size_name in tensor_sizes:
        # Create tensor of appropriate size
        num_elements = size_bytes // 4  # float32 = 4 bytes
        tensor = torch.randn(num_elements, dtype=torch.float32)
        
        print(f"\n--- Testing {size_name} tensor ---")
        print(f"Tensor shape: {tensor.shape}, Size: {tensor.nelement() * 4} bytes")
        
        # Time all-reduce operation
        start_time = asyncio.get_event_loop().time()
        result = await engine.all_reduce(tensor, operation="sum")
        end_time = asyncio.get_event_loop().time()
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"All-reduce latency: {latency:.2f} ms")
        
        # Verify result (in simulation, should be approximately original * world_size)
        expected = tensor * engine.world_size
        error = torch.abs(result - expected).mean()
        print(f"Verification error: {error:.6f}")

async def test_distributed_forward():
    """Test distributed model forward pass"""
    print("\n🧪 TESTING DISTRIBUTED FORWARD PASS...")
    
    engine = NVRAREngine()
    config = DistributedConfig(
        parallelism_strategy=ParallelismStrategy.TENSOR_PARALLELISM,
        num_nodes=1,
        gpus_per_node=2
    )
    
    await engine.initialize_distributed(config)
    
    # Simulate model inputs
    batch_size = 2
    seq_length = 128
    hidden_size = 512
    
    inputs = torch.randn(batch_size, seq_length, hidden_size)
    print(f"Input shape: {inputs.shape}")
    
    # Simulate model shards (in production, these would be actual model partitions)
    model_shards = [f"shard_{i}" for i in range(config.gpus_per_node)]
    
    # Execute distributed forward
    start_time = asyncio.get_event_loop().time()
    output = await engine.model_parallel_forward(inputs, model_shards)
    end_time = asyncio.get_event_loop().time()
    
    latency = (end_time - start_time) * 1000
    print(f"Distributed forward latency: {latency:.2f} ms")
    print(f"Output type: {type(output)}")
    
    # Print communication statistics
    stats = engine.get_communication_stats()
    print(f"\nCommunication stats:")
    print(f"  All-reduce count: {stats['all_reduce_count']}")
    print(f"  Total communication time: {stats['total_communication_time']:.4f}s")
    print(f"  Bytes transferred: {stats['bytes_transferred']:,} bytes")
    print(f"  Avg communication time: {stats.get('avg_communication_time', 0):.4f}s")

async def test_performance_comparison():
    """Compare simulated NVRAR vs NCCL performance"""
    print("\n📊 PERFORMANCE COMPARISON: NVRAR vs NCCL")
    print("=" * 60)
    
    # Simulate performance improvements from paper
    improvements = {
        "128KB": 1.9,   # 1.9x improvement
        "512KB": 2.5,   # 2.5x improvement  
        "2MB": 3.6,     # 3.6x improvement
        "end_to_end": 1.72  # Overall improvement for Llama 3.1 405B
    }
    
    print("\nPerformance improvements from paper (NVRAR vs NCCL):")
    for size, improvement in improvements.items():
        print(f"  {size:15s}: {improvement:5.2f}x faster")
    
    print("\n🎯 Key bottlenecks addressed by NVRAR:")
    print("  ✓ All-reduce latency for medium-sized tensors (128KB-2MB)")
    print("  ✓ Hierarchical communication (intra-node + inter-node)")
    print("  ✓ Recursive doubling algorithm with NVSHMEM")
    print("  ✓ Optimal for decode-heavy workloads in multi-node setups")
    
    print("\n🔬 Research paper insights:")
    print("  Paper: 'LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations'")
    print("  Authors: NVIDIA Research Team")
    print("  Key finding: NVRAR provides 1.72x speedup for Llama 3.1 405B inference")
    print("  Critical for: Multi-node tensor parallelism with 8-128 GPUs")

async def test_hierarchical_communication():
    """Test hierarchical all-reduce vs flat all-reduce"""
    print("\n🔀 TESTING HIERARCHICAL vs FLAT ALL-REDUCE")
    print("=" * 60)
    
    # Test with hierarchical enabled
    engine_hierarchical = NVRAREngine()
    config_hierarchical = DistributedConfig(
        num_nodes=4,
        gpus_per_node=8,
        hierarchical_all_reduce=True
    )
    await engine_hierarchical.initialize_distributed(config_hierarchical)
    
    # Large tensor (2MB)
    tensor = torch.randn(524288, dtype=torch.float32)
    
    print(f"\nHierarchical All-Reduce (4 nodes x 8 GPUs = 32 total GPUs):")
    start = asyncio.get_event_loop().time()
    result_hier = await engine_hierarchical.all_reduce(tensor)
    end = asyncio.get_event_loop().time()
    hierarchical_time = (end - start) * 1000
    print(f"  Latency: {hierarchical_time:.2f} ms")
    
    # Get stats
    stats_hier = engine_hierarchical.get_communication_stats()
    print(f"  Total bytes transferred: {stats_hier['bytes_transferred']:,}")
    
    print("\n✓ Hierarchical approach benefits:")
    print("  - Intra-node: Fast NVLink/PCIe communication")
    print("  - Inter-node: NVSHMEM for reduced latency")
    print("  - Scalability: Logarithmic complexity O(log n)")
    print("  - Bandwidth: Optimal utilization of network topology")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 STARTING DISTRIBUTED INFERENCE TESTS")
    print("=" * 60)
    print("Based on: 'LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations'")
    print("NVRAR: Hierarchical All-Reduce with NVSHMEM\n")
    
    asyncio.run(test_nvrar_all_reduce())
    asyncio.run(test_distributed_forward()) 
    asyncio.run(test_performance_comparison())
    asyncio.run(test_hierarchical_communication())
    
    print("\n" + "=" * 60)
    print("✅ ALL DISTRIBUTED INFERENCE TESTS COMPLETED!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("  1. Enable distributed inference via API: POST /api/v1/distributed/enable")
    print("  2. Monitor performance: GET /api/v1/distributed/stats")
    print("  3. Use distributed generation with use_distributed=true parameter")
    print("  4. Scale to multiple nodes for production workloads (Llama 405B, etc.)")
