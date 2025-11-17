#!/bin/bash
# Scalability benchmark script - tests different batch sizes
# Usage: ./benchmark_scalability.sh [--runs N] [--batch_sizes "1,2,4,8,16"]

# Default values
RUNS=10
BATCH_SIZES="1,2,4,8,16"
MODEL_DIR="models"
OUTPUT_DIR="results/scalability"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --batch_sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --model)
            MODEL_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--runs N] [--batch_sizes \"1,2,4,8,16\"] [--model DIR] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if executables exist
if [ ! -f "build/cnn_inference_cpu" ]; then
    echo "Error: build/cnn_inference_cpu not found. Run 'make cpu' first."
    exit 1
fi

if [ ! -f "build/cnn_inference_cpu_multicore" ]; then
    echo "Error: build/cnn_inference_cpu_multicore not found. Run 'make cpu_multicore' first."
    exit 1
fi

if [ ! -f "build/cnn_inference_gpu" ]; then
    echo "Error: build/cnn_inference_gpu not found. Run 'make gpu' first."
    exit 1
fi

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory '$MODEL_DIR' not found."
    exit 1
fi

echo "=== Scalability Benchmark ==="
echo "Runs per batch size: $RUNS"
echo "Batch sizes: $BATCH_SIZES"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Convert comma-separated batch sizes to array
IFS=',' read -ra BATCH_ARRAY <<< "$BATCH_SIZES"

# Run benchmarks for each batch size
for BATCH in "${BATCH_ARRAY[@]}"; do
    echo "=== Batch Size: $BATCH ==="
    
    # CPU Sequential
    echo "Running CPU Sequential..."
    ./benchmark_cpu.sh \
        --runs "$RUNS" \
        --batch "$BATCH" \
        --model "$MODEL_DIR" \
        --output "$OUTPUT_DIR/cpu_batch${BATCH}.csv"
    
    # CPU Multicore
    echo "Running CPU Multicore..."
    ./benchmark_cpu_multicore.sh \
        --runs "$RUNS" \
        --batch "$BATCH" \
        --model "$MODEL_DIR" \
        --output "$OUTPUT_DIR/cpu_multicore_batch${BATCH}.csv"
    
    # GPU
    echo "Running GPU CUDA..."
    ./benchmark_gpu.sh \
        --runs "$RUNS" \
        --batch "$BATCH" \
        --model "$MODEL_DIR" \
        --output "$OUTPUT_DIR/gpu_batch${BATCH}.csv"
    
    echo ""
done

echo "=== Scalability Benchmark Complete ==="
echo "Results saved to: $OUTPUT_DIR/"

