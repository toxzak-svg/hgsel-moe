#!/usr/bin/env python
"""Performance report generator for HGSEL benchmarks.

Parses benchmark results and generates:
- Markdown report with analysis
- Matplotlib plots for scaling curves
- Memory and latency breakdowns
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate HGSEL Performance Report")
    parser.add_argument("--results", type=str, default="results/benchmark/benchmark_results.json",
                       help="Path to benchmark results JSON")
    parser.add_argument("--output", type=str, default="results/benchmark",
                       help="Output directory for report")
    parser.add_argument("--include-plots", action="store_true", help="Generate matplotlib plots")
    return parser.parse_args()


def load_results(results_file: str) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def analyze_throughput(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze throughput results."""
    analysis = {
        "configurations": len(results),
        "by_batch_size": {},
        "by_expert_count": {},
        "peak_throughput": 0,
        "peak_config": None,
    }
    
    for result in results:
        batch_size = result["batch_size"]
        num_experts = result["num_experts"]
        throughput = result["throughput"]["tokens_per_sec"]
        
        # By batch size
        if batch_size not in analysis["by_batch_size"]:
            analysis["by_batch_size"][batch_size] = []
        analysis["by_batch_size"][batch_size].append({
            "num_experts": num_experts,
            "throughput": throughput,
        })
        
        # By expert count
        if num_experts not in analysis["by_expert_count"]:
            analysis["by_expert_count"][num_experts] = []
        analysis["by_expert_count"][num_experts].append({
            "batch_size": batch_size,
            "throughput": throughput,
        })
        
        # Peak
        if throughput > analysis["peak_throughput"]:
            analysis["peak_throughput"] = throughput
            analysis["peak_config"] = {
                "batch_size": batch_size,
                "num_experts": num_experts,
            }
    
    return analysis


def analyze_memory(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze memory results."""
    memory_results = [r for r in results if "memory" in r]
    
    if not memory_results:
        return {"available": False, "reason": "No memory profiling data"}
    
    analysis = {
        "available": True,
        "peak_memory_mb": 0,
        "peak_config": None,
        "memory_by_config": [],
    }
    
    for result in memory_results:
        batch_size = result["batch_size"]
        num_experts = result["num_experts"]
        
        try:
            snapshots = result["memory"]["snapshots"]
            memory_mb = max(s["allocated_mb"] for s in snapshots)
            
            analysis["memory_by_config"].append({
                "batch_size": batch_size,
                "num_experts": num_experts,
                "memory_mb": memory_mb,
            })
            
            if memory_mb > analysis["peak_memory_mb"]:
                analysis["peak_memory_mb"] = memory_mb
                analysis["peak_config"] = {
                    "batch_size": batch_size,
                    "num_experts": num_experts,
                }
        except (KeyError, ValueError):
            pass
    
    return analysis


def analyze_latency(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze latency results."""
    latency_results = [r for r in results if "latency" in r]
    
    if not latency_results:
        return {"available": False, "reason": "No latency profiling data"}
    
    analysis = {
        "available": True,
        "latency_configs": [],
    }
    
    for result in latency_results:
        batch_size = result["batch_size"]
        num_experts = result["num_experts"]
        
        try:
            latency_data = result["latency"]
            analysis["latency_configs"].append({
                "batch_size": batch_size,
                "num_experts": num_experts,
                "p50_ms": latency_data.get("p50_ms", 0),
                "p99_ms": latency_data.get("p99_ms", 0),
                "mean_ms": latency_data.get("mean_ms", 0),
            })
        except (KeyError, ValueError):
            pass
    
    return analysis


def generate_markdown_report(
    results: List[Dict[str, Any]],
    throughput_analysis: Dict,
    memory_analysis: Dict,
    latency_analysis: Dict,
) -> str:
    """Generate markdown report."""
    report = []
    report.append("# HGSEL 300M Performance Report\n")
    
    # Summary
    report.append("## Summary\n")
    report.append(f"- Total configurations: {len(results)}\n")
    report.append(f"- Peak throughput: {throughput_analysis['peak_throughput']:.1f} tokens/sec\n")
    if throughput_analysis["peak_config"]:
        cfg = throughput_analysis["peak_config"]
        report.append(f"  - Configuration: batch_size={cfg['batch_size']}, num_experts={cfg['num_experts']}\n")
    
    if memory_analysis["available"]:
        report.append(f"- Peak memory: {memory_analysis['peak_memory_mb']:.1f} MB\n")
    
    report.append("\n")
    
    # Throughput Analysis
    report.append("## Throughput Analysis\n")
    report.append("### By Batch Size\n")
    for batch_size in sorted(throughput_analysis["by_batch_size"].keys()):
        entries = throughput_analysis["by_batch_size"][batch_size]
        avg_throughput = np.mean([e["throughput"] for e in entries])
        report.append(f"- Batch size {batch_size}: {avg_throughput:.1f} tokens/sec (avg)\n")
    
    report.append("\n### By Expert Count\n")
    for num_experts in sorted(throughput_analysis["by_expert_count"].keys()):
        entries = throughput_analysis["by_expert_count"][num_experts]
        avg_throughput = np.mean([e["throughput"] for e in entries])
        report.append(f"- {num_experts} experts: {avg_throughput:.1f} tokens/sec (avg)\n")
    
    report.append("\n")
    
    # Memory Analysis
    if memory_analysis["available"]:
        report.append("## Memory Analysis\n")
        for config in memory_analysis["memory_by_config"]:
            report.append(f"- B={config['batch_size']}, E={config['num_experts']}: "
                         f"{config['memory_mb']:.1f} MB\n")
        report.append("\n")
    
    # Latency Analysis
    if latency_analysis["available"]:
        report.append("## Latency Analysis\n")
        for config in latency_analysis["latency_configs"]:
            report.append(f"- B={config['batch_size']}, E={config['num_experts']}: "
                         f"p50={config['p50_ms']:.2f}ms, p99={config['p99_ms']:.2f}ms\n")
        report.append("\n")
    
    # Recommendations
    report.append("## Recommendations\n")
    
    # Find best config for throughput
    best_config = throughput_analysis["peak_config"]
    if best_config:
        report.append(f"1. **Optimal Configuration**: batch_size={best_config['batch_size']}, "
                     f"num_experts={best_config['num_experts']}\n")
    
    # Memory scaling
    if memory_analysis["available"] and len(memory_analysis["memory_by_config"]) > 1:
        report.append("2. **Memory Scaling**: Consider batch size trade-offs - smaller batches "
                     "reduce memory overhead\n")
    
    report.append("3. **Next Steps**: Profile communication overhead and consider overlapped dispatch\n")
    
    return "".join(report)


def generate_plots(results: List[Dict[str, Any]], output_dir: Path):
    """Generate matplotlib plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    # Throughput by batch size
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("HGSEL 300M Performance Analysis")
    
    # Extract data
    batch_sizes_set = set(r["batch_size"] for r in results)
    expert_counts_set = set(r["num_experts"] for r in results)
    
    # Plot 1: Throughput by batch size
    ax = axes[0, 0]
    for num_experts in sorted(expert_counts_set):
        x_vals = []
        y_vals = []
        for result in results:
            if result["num_experts"] == num_experts:
                x_vals.append(result["batch_size"])
                y_vals.append(result["throughput"]["tokens_per_sec"])
        if x_vals:
            ax.plot(sorted(zip(x_vals, y_vals)), marker='o', label=f"E={num_experts}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Batch Size")
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Throughput by expert count
    ax = axes[0, 1]
    for batch_size in sorted(batch_sizes_set):
        x_vals = []
        y_vals = []
        for result in results:
            if result["batch_size"] == batch_size:
                x_vals.append(result["num_experts"])
                y_vals.append(result["throughput"]["tokens_per_sec"])
        if x_vals:
            ax.plot(sorted(zip(x_vals, y_vals)), marker='s', label=f"B={batch_size}")
    ax.set_xlabel("Number of Experts")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Expert Count")
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Memory (if available)
    ax = axes[1, 0]
    memory_results = [r for r in results if "memory" in r]
    if memory_results:
        configs = [f"B={r['batch_size']},E={r['num_experts']}" for r in memory_results]
        memory_mb = [max(s["allocated_mb"] for s in r["memory"]["snapshots"])
                    for r in memory_results]
        ax.bar(range(len(configs)), memory_mb)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage by Configuration")
    else:
        ax.text(0.5, 0.5, "No memory data", ha='center', va='center')
    ax.grid(True, axis='y')
    
    # Plot 4: Latency (if available)
    ax = axes[1, 1]
    latency_results = [r for r in results if "latency" in r]
    if latency_results:
        configs = [f"B={r['batch_size']},E={r['num_experts']}" for r in latency_results]
        p50_ms = [r["latency"].get("p50_ms", 0) for r in latency_results]
        p99_ms = [r["latency"].get("p99_ms", 0) for r in latency_results]
        
        x = np.arange(len(configs))
        width = 0.35
        ax.bar(x - width/2, p50_ms, width, label="p50")
        ax.bar(x + width/2, p99_ms, width, label="p99")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency by Configuration")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No latency data", ha='center', va='center')
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_analysis.png", dpi=150)
    print(f"Plots saved to {output_dir}/performance_analysis.png")


def main():
    """Main entry point."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return 1
    
    print(f"Loading results from {results_file}")
    results = load_results(str(results_file))
    print(f"Loaded {len(results)} results")
    
    # Analyze
    throughput_analysis = analyze_throughput(results)
    memory_analysis = analyze_memory(results)
    latency_analysis = analyze_latency(results)
    
    # Generate report
    report = generate_markdown_report(results, throughput_analysis, memory_analysis, latency_analysis)
    report_file = output_dir / "PERFORMANCE_REPORT.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to {report_file}")
    
    # Generate plots
    if args.include_plots:
        generate_plots(results, output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())
