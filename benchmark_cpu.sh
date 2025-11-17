#!/bin/bash
# Benchmark script for CPU sequential implementation
# Usage: ./benchmark_cpu.sh [--runs N] [--output FILE] [--batch N]

# Default values
RUNS=10
OUTPUT="results/cpu_baseline.csv"
BATCH=1
MODEL_DIR="models"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --model)
            MODEL_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--runs N] [--output FILE] [--batch N] [--model DIR]"
            exit 1
            ;;
    esac
done

# Create results directory
mkdir -p "$(dirname "$OUTPUT")"

# Check if executable exists
if [ ! -f "build/cnn_inference_cpu" ]; then
    echo "Error: build/cnn_inference_cpu not found. Run 'make cpu' first."
    exit 1
fi

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory '$MODEL_DIR' not found."
    exit 1
fi

echo "=== CPU Sequential Benchmark ==="
echo "Runs: $RUNS"
echo "Batch size: $BATCH"
echo "Output: $OUTPUT"
echo ""

# Write CSV header
echo "run,batch_size,time_ms,memory_mb" > "$OUTPUT"

# Run benchmark
for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS..."
    
    # Run inference and capture output
    # Note: Current implementation doesn't support batch size yet
    # This will be updated when batch inference is implemented
    
    # Measure execution time
    START=$(date +%s.%N 2>/dev/null || date +%s)
    OUTPUT_TEXT=$(./build/cnn_inference_cpu --model "$MODEL_DIR" --device cpu 2>&1)
    END=$(date +%s.%N 2>/dev/null || date +%s)
    
    # Extract time from output (if available)
    TIME_MS=$(echo "$OUTPUT_TEXT" | grep "Inference completed" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    # If time extraction failed, use wall-clock time
    if [ -z "$TIME_MS" ]; then
        # Use awk for floating point calculation (more portable than bc)
        ELAPSED=$(awk "BEGIN {print $END - $START}")
        TIME_MS=$(awk "BEGIN {print $ELAPSED * 1000}")
    fi
    
    # Measure memory usage (peak RSS in MB)
    # Note: This is approximate - actual memory profiling requires valgrind or similar
    MEMORY_MB=$(ps aux 2>/dev/null | grep cnn_inference_cpu | grep -v grep | awk '{print $6/1024}' | head -1)
    if [ -z "$MEMORY_MB" ] || [ "$MEMORY_MB" = "0" ]; then
        MEMORY_MB="N/A"
    fi
    
    # Write to CSV
    echo "$i,$BATCH,$TIME_MS,$MEMORY_MB" >> "$OUTPUT"
    
    # Print progress
    echo "  Time: $TIME_MS ms, Memory: $MEMORY_MB MB"
done

echo ""
echo "Benchmark complete. Results saved to: $OUTPUT"
echo ""
echo "Summary:"
# Calculate summary statistics using awk
awk -F',' '
NR>1 && $3 != "N/A" && $3 != "" {
    sum+=$3; 
    count++; 
    if(min=="" || $3<min) min=$3; 
    if($3>max) max=$3
} 
END {
    if(count>0) {
        mean = sum/count
        printf "  Runs: %d\n  Mean: %.2f ms\n  Min: %.2f ms\n  Max: %.2f ms\n", count, mean, min, max
    } else {
        print "  No valid time measurements found"
    }
}' "$OUTPUT"

