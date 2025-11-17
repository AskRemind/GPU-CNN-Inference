#!/usr/bin/env python3
"""
Performance evaluation script for CNN inference results.

Evaluates:
1. Speedup: Compare different implementations
2. Scalability: Performance with varying batch sizes and resolutions
3. Memory Usage: Peak memory consumption

Usage:
    python3 evaluate_performance.py --input results.csv --output analysis.csv
    python3 evaluate_performance.py --input results.csv --plot output_dir/
"""

import csv
import argparse
import numpy as np
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")


def load_csv(filename):
    """Load results from CSV file."""
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns
            for key in ['run', 'batch_size', 'time_ms', 'memory_mb']:
                if key in row and row[key] and row[key] != 'N/A':
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            results.append(row)
    return results


def calculate_statistics(results):
    """Calculate statistics for results."""
    times = [r['time_ms'] for r in results if r['time_ms'] is not None]
    memories = [r['memory_mb'] for r in results if r['memory_mb'] is not None]
    
    stats = {}
    if times:
        stats['time'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    if memories:
        stats['memory'] = {
            'mean': np.mean(memories),
            'std': np.std(memories),
            'min': np.min(memories),
            'max': np.max(memories),
            'median': np.median(memories)
        }
    
    return stats


def evaluate_speedup(baseline_stats, other_stats):
    """Calculate speedup compared to baseline."""
    if baseline_stats.get('time') and other_stats.get('time'):
        baseline_mean = baseline_stats['time']['mean']
        other_mean = other_stats['time']['mean']
        if other_mean > 0:
            speedup = baseline_mean / other_mean
            return speedup
    return None


def evaluate_scalability(results_by_batch):
    """Evaluate scalability across batch sizes."""
    scalability = {}
    
    for batch_size, results in sorted(results_by_batch.items()):
        stats = calculate_statistics(results)
        if stats.get('time'):
            # Calculate throughput (images per second)
            throughput = 1000.0 / stats['time']['mean']  # ms to images/sec
            scalability[batch_size] = {
                'mean_time_ms': stats['time']['mean'],
                'throughput_ips': throughput,
                'memory_mb': stats['memory']['mean'] if stats.get('memory') else None
            }
    
    return scalability


def evaluate_memory_usage(results):
    """Evaluate memory usage patterns."""
    stats = calculate_statistics(results)
    if stats.get('memory'):
        return {
            'peak_mb': stats['memory']['max'],
            'mean_mb': stats['memory']['mean'],
            'std_mb': stats['memory']['std']
        }
    return None


def plot_performance(results, output_dir):
    """Generate performance plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots: matplotlib not available")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by batch size
    results_by_batch = {}
    for r in results:
        batch = r.get('batch_size', 1)
        if batch not in results_by_batch:
            results_by_batch[batch] = []
        results_by_batch[batch].append(r)
    
    # Plot 1: Time vs Batch Size
    if len(results_by_batch) > 1:
        batches = sorted(results_by_batch.keys())
        mean_times = []
        std_times = []
        
        for batch in batches:
            stats = calculate_statistics(results_by_batch[batch])
            if stats.get('time'):
                mean_times.append(stats['time']['mean'])
                std_times.append(stats['time']['std'])
            else:
                mean_times.append(0)
                std_times.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(batches, mean_times, yerr=std_times, marker='o', capsize=5)
        plt.xlabel('Batch Size')
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time vs Batch Size')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'time_vs_batch.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Throughput vs Batch Size
    if len(results_by_batch) > 1:
        batches = sorted(results_by_batch.keys())
        throughputs = []
        
        for batch in batches:
            stats = calculate_statistics(results_by_batch[batch])
            if stats.get('time'):
                throughput = 1000.0 / stats['time']['mean']
                throughputs.append(throughput)
            else:
                throughputs.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(batches, throughputs, marker='o', linewidth=2)
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (images/sec)')
        plt.title('Throughput vs Batch Size')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'throughput_vs_batch.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Memory Usage vs Batch Size
    if len(results_by_batch) > 1:
        batches = sorted(results_by_batch.keys())
        memories = []
        
        for batch in batches:
            stats = calculate_statistics(results_by_batch[batch])
            if stats.get('memory'):
                memories.append(stats['memory']['mean'])
            else:
                memories.append(0)
        
        if any(m > 0 for m in memories):
            plt.figure(figsize=(10, 6))
            plt.plot(batches, memories, marker='o', linewidth=2, color='orange')
            plt.xlabel('Batch Size')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage vs Batch Size')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / 'memory_vs_batch.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Plots saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN inference performance')
    parser.add_argument('--input', required=True, help='Input CSV file with results')
    parser.add_argument('--output', help='Output CSV file for analysis')
    parser.add_argument('--plot', help='Output directory for plots')
    parser.add_argument('--baseline', help='Baseline CSV file for speedup calculation')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    results = load_csv(args.input)
    
    if not results:
        print("Error: No results found in input file")
        return
    
    print(f"Loaded {len(results)} results")
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    print("\n=== Performance Statistics ===")
    if stats.get('time'):
        print(f"Time (ms):")
        print(f"  Mean: {stats['time']['mean']:.2f}")
        print(f"  Std:  {stats['time']['std']:.2f}")
        print(f"  Min:  {stats['time']['min']:.2f}")
        print(f"  Max:  {stats['time']['max']:.2f}")
        print(f"  Median: {stats['time']['median']:.2f}")
    
    if stats.get('memory'):
        print(f"\nMemory (MB):")
        print(f"  Mean: {stats['memory']['mean']:.2f}")
        print(f"  Std:  {stats['memory']['std']:.2f}")
        print(f"  Min:  {stats['memory']['min']:.2f}")
        print(f"  Max:  {stats['memory']['max']:.2f}")
    
    # Evaluate scalability
    results_by_batch = {}
    for r in results:
        batch = r.get('batch_size', 1)
        if batch not in results_by_batch:
            results_by_batch[batch] = []
        results_by_batch[batch].append(r)
    
    scalability = None
    if len(results_by_batch) > 1:
        scalability = evaluate_scalability(results_by_batch)
        print("\n=== Scalability Analysis ===")
        for batch, metrics in sorted(scalability.items()):
            print(f"Batch Size {batch}:")
            print(f"  Mean Time: {metrics['mean_time_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput_ips']:.2f} images/sec")
            if metrics['memory_mb']:
                print(f"  Memory: {metrics['memory_mb']:.2f} MB")
    
    # Evaluate speedup if baseline provided
    if args.baseline:
        print("\n=== Speedup Analysis ===")
        baseline_results = load_csv(args.baseline)
        baseline_stats = calculate_statistics(baseline_results)
        current_stats = stats
        
        speedup = evaluate_speedup(baseline_stats, current_stats)
        if speedup:
            print(f"Speedup vs Baseline: {speedup:.2f}x")
            if speedup > 1.0:
                print(f"  {speedup:.1f}x faster than baseline")
            else:
                print(f"  {1.0/speedup:.1f}x slower than baseline")
    
    # Memory usage analysis
    memory_usage = evaluate_memory_usage(results)
    if memory_usage:
        print("\n=== Memory Usage Analysis ===")
        print(f"Peak Memory: {memory_usage['peak_mb']:.2f} MB")
        print(f"Mean Memory: {memory_usage['mean_mb']:.2f} MB")
        print(f"Std Memory: {memory_usage['std_mb']:.2f} MB")
    
    # Save analysis to JSON
    if args.output:
        output_data = {
            'statistics': stats,
            'scalability': scalability,
            'memory_usage': memory_usage
        }
        
        # Save as JSON for easier parsing
        with open(args.output.replace('.csv', '.json'), 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nAnalysis saved to {args.output.replace('.csv', '.json')}")
    
    # Generate plots
    if args.plot:
        print("\n=== Generating Plots ===")
        plot_performance(results, args.plot)


if __name__ == '__main__':
    main()

