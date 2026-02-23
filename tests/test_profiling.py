#!/usr/bin/env python
"""
Tests for Phase 5 profiling and benchmarking infrastructure.

Tests verify:
- Overlapped dispatch correctness (outputs match sequential)
- Memory profiler accuracy
- Throughput calculation validity
- Latency decomposition components
- Scaling behavior across configurations
"""

import unittest
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from hgsel.distributed.overlapped_dispatch import OverlappedDispatchPipeline, OverlapMetrics
from hgsel.distributed.memory_profiler import MemoryProfiler, estimate_model_memory_requirements
from hgsel.distributed.throughput_benchmark import ThroughputBenchmark, ThroughputMetrics
from hgsel.distributed.latency_profiler import LatencyProfiler, LatencyBreakdown
from hgsel.training.data import DummyDataLoader
from experiments.baselines.dense_transformer import TransformerModel


class TestOverlappedDispatch(unittest.TestCase):
    """Test overlapped dispatch pipeline."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 128
        self.batch_size = 16
        self.seq_length = 64
    
    def test_overlapped_dispatch_output_correctness(self):
        """Verify overlapped dispatch produces same output as sequential."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
        ).to(self.device)
        
        # Create pipeline
        pipeline = OverlappedDispatchPipeline(model, device=self.device)
        
        # Create test input
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_size,
                       device=self.device)
        
        # Sequential forward
        with torch.no_grad():
            sequential_out = model(x)
        
        # Overlapped forward
        with torch.no_grad():
            overlapped_out, metrics = pipeline._forward_overlapped(x)
        
        # Outputs should match (within numerical precision)
        torch.testing.assert_close(sequential_out, overlapped_out, atol=1e-5, rtol=1e-5)
    
    def test_overlap_metrics_valid(self):
        """Verify overlap metrics are reasonable."""
        model = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        pipeline = OverlappedDispatchPipeline(model, device=self.device)
        
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_size,
                       device=self.device)
        
        _, metrics = pipeline._forward_overlapped(x)
        
        # Check metrics structure
        self.assertIsInstance(metrics, OverlapMetrics)
        self.assertGreater(metrics.local_compute_ms, 0)
        
        # Overlap ratio should be in [0, 1]
        if metrics.exchange_ms > 0:
            overlap_ratio = metrics.local_compute_ms / metrics.exchange_ms
            self.assertGreaterEqual(overlap_ratio, 0)


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiling."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.Linear(512, 256),
        ).to(self.device)
    
    def test_memory_profiler_tracks_snapshots(self):
        """Verify memory profiler records snapshots."""
        profiler = MemoryProfiler(self.model)
        
        # Take initial snapshot
        profiler.take_snapshot("start")
        self.assertEqual(len(profiler.snapshots), 1)
        
        # Run some computation
        x = torch.randn(32, 256, device=self.device)
        y = self.model(x)
        
        # Take another snapshot
        profiler.take_snapshot("after_forward")
        self.assertEqual(len(profiler.snapshots), 2)
        
        # Snapshots should have reasonable data
        for snap in profiler.snapshots:
            self.assertGreater(snap["allocated_mb"], 0)
            self.assertGreater(snap["reserved_mb"], 0)
            self.assertGreater(snap["timestamp"], 0)
    
    def test_memory_estimation(self):
        """Verify memory estimation for models."""
        # Estimate memory requirements
        mem_reqs = estimate_model_memory_requirements(self.model)
        
        # Check structure
        self.assertIn("total_params", mem_reqs)
        self.assertIn("param_memory_mb", mem_reqs)
        self.assertIn("gradient_memory_mb", mem_reqs)
        
        # Values should be positive
        self.assertGreater(mem_reqs["total_params"], 0)
        self.assertGreater(mem_reqs["param_memory_mb"], 0)
        self.assertGreater(mem_reqs["gradient_memory_mb"], 0)


class TestThroughputBenchmark(unittest.TestCase):
    """Test throughput measurement."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerModel(
            vocab_size=256,
            d_model=128,
            num_heads=2,
            d_ff=256,
            num_layers=2,
            max_seq_length=64,
        ).to(self.device)
    
    def test_throughput_metrics_valid(self):
        """Verify throughput metrics are valid."""
        benchmark = ThroughputBenchmark(self.model, device=self.device)
        
        # Create minimal data loader
        data_loader = DummyDataLoader(
            num_batches=5,
            batch_size=16,
            seq_length=64,
            vocab_size=256,
        )
        
        metrics = benchmark.run(
            data_loader,
            num_warmup_steps=1,
            num_bench_steps=3,
        )
        
        # Verify metrics
        self.assertIsInstance(metrics, ThroughputMetrics)
        self.assertGreater(metrics.tokens_per_sec, 0)
        self.assertGreater(metrics.total_time_sec, 0)
        self.assertLessEqual(metrics.utilization_percent, 100)
        self.assertGreaterEqual(metrics.utilization_percent, 0)
    
    def test_throughput_calculation(self):
        """Verify throughput calculation is correct."""
        benchmark = ThroughputBenchmark(self.model, device=self.device)
        
        # Manually check calculation
        batch_size = 16
        seq_length = 64
        num_steps = 3
        
        expected_tokens = batch_size * seq_length * num_steps
        
        data_loader = DummyDataLoader(
            num_batches=5,
            batch_size=batch_size,
            seq_length=seq_length,
            vocab_size=256,
        )
        
        metrics = benchmark.run(data_loader, num_warmup_steps=1, num_bench_steps=num_steps)
        
        # Tokens per second * time should give total tokens (approx)
        calculated_tokens = metrics.tokens_per_sec * metrics.total_time_sec
        self.assertAlmostEqual(calculated_tokens, expected_tokens, delta=expected_tokens * 0.1)


class TestLatencyProfiler(unittest.TestCase):
    """Test latency profiling."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
    
    def test_latency_breakdown_complete(self):
        """Verify latency breakdown has all components."""
        profiler = LatencyProfiler(self.model, self.optimizer, device=self.device)
        
        # Create test batch
        batch = torch.randn(16, 256, device=self.device)
        
        # Profile step
        breakdown = profiler.profile_step(batch)
        
        # Verify structure
        self.assertIsInstance(breakdown, LatencyBreakdown)
        self.assertGreater(breakdown.forward_ms, 0)
        self.assertGreater(breakdown.total_ms, 0)
        
        # Components should sum to approximately total
        component_sum = (
            breakdown.forward_ms + breakdown.backward_ms +
            breakdown.all_to_all_ms + breakdown.all_reduce_ms +
            breakdown.optimizer_ms + breakdown.synchronize_ms +
            breakdown.other_ms
        )
        self.assertAlmostEqual(component_sum, breakdown.total_ms, delta=breakdown.total_ms * 0.2)
    
    def test_latency_percentiles(self):
        """Verify latency statistics computation."""
        profiler = LatencyProfiler(self.model, self.optimizer, device=self.device)
        
        # Profile multiple steps
        for _ in range(10):
            batch = torch.randn(16, 256, device=self.device)
            profiler.profile_step(batch)
        
        # Get statistics
        stats = profiler.stats()
        
        # Verify percentiles are ordered
        self.assertLess(stats.p50_ms, stats.p99_ms)
        self.assertLess(stats.p99_ms, stats.p999_ms)
        self.assertGreater(stats.mean_ms, 0)
        self.assertGreater(stats.std_ms, 0)


class TestPhase5Integration(unittest.TestCase):
    """Integration tests for Phase 5 components."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerModel(
            vocab_size=256,
            d_model=128,
            num_heads=2,
            d_ff=256,
            num_layers=2,
            max_seq_length=64,
        ).to(self.device)
    
    def test_profilers_work_together(self):
        """Verify all profilers can run on same model."""
        # Create profilers
        memory_profiler = MemoryProfiler(self.model)
        benchmark = ThroughputBenchmark(self.model, device=self.device)
        latency_profiler = LatencyProfiler(
            self.model,
            torch.optim.Adam(self.model.parameters()),
            device=self.device
        )
        
        # Run all profilers on same model
        data_loader = DummyDataLoader(num_batches=5, batch_size=16, seq_length=64, vocab_size=256)
        
        memory_profiler.take_snapshot("start")
        metrics = benchmark.run(data_loader, num_warmup_steps=1, num_bench_steps=2)
        memory_profiler.take_snapshot("end")
        
        # All should complete successfully
        self.assertGreater(metrics.tokens_per_sec, 0)
        self.assertEqual(len(memory_profiler.snapshots), 2)
    
    def test_memory_scales_with_batch_size(self):
        """Verify memory increases with batch size."""
        configs = [
            (8, 64),
            (16, 64),
            (32, 64),
        ]
        
        memory_usage = []
        
        for batch_size, seq_len in configs:
            model = TransformerModel(
                vocab_size=256,
                d_model=128,
                num_heads=2,
                d_ff=256,
                num_layers=2,
                max_seq_length=seq_len,
            ).to(self.device)
            
            profiler = MemoryProfiler(model)
            profiler.take_snapshot("start")
            
            # Forward pass
            x = torch.randn(batch_size, seq_len, device=self.device, dtype=torch.long)
            _ = model(x[:, :, 0].long())  # Extract token IDs
            
            profiler.take_snapshot("forward")
            mem = profiler.snapshots[-1]["allocated_mb"]
            memory_usage.append((batch_size, mem))
        
        # Verify memory increases with batch size
        for i in range(1, len(memory_usage)):
            self.assertGreater(
                memory_usage[i][1],
                memory_usage[i-1][1],
                f"Memory should increase with batch size"
            )


if __name__ == "__main__":
    # Run tests
    unittest.main()
