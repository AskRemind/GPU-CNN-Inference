#!/bin/bash
# GPU profiling script using nvprof
# Usage: ./benchmark_gpu_profiling.sh [--batch_size N] [--output_dir DIR]

# Default values
BATCH_SIZE=16
MODEL_DIR="models"
OUTPUT_DIR="results/profiling"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
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
            echo "Usage: $0 [--batch_size N] [--model DIR] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if executable exists
if [ ! -f "build/cnn_inference_gpu" ]; then
    echo "Error: build/cnn_inference_gpu not found. Run 'make gpu' first."
    exit 1
fi

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory '$MODEL_DIR' not found."
    exit 1
fi

# Check if nvprof is available
if ! command -v nvprof >/dev/null 2>&1; then
    echo "Error: nvprof not found. Cannot run profiling."
    exit 1
fi

echo "=== GPU Profiling with nvprof ==="
echo "Batch size: $BATCH_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Output file names
PROF_OUTPUT="$OUTPUT_DIR/gpu_profiling_batch${BATCH_SIZE}.txt"
CSV_OUTPUT="$OUTPUT_DIR/gpu_profiling_batch${BATCH_SIZE}.csv"
SUMMARY_OUTPUT="$OUTPUT_DIR/gpu_profiling_batch${BATCH_SIZE}_summary.txt"

# Run nvprof profiling
echo "Running nvprof profiling..."
nvprof \
    --output-profile "$CSV_OUTPUT" \
    --print-gpu-trace \
    --print-api-trace \
    --log-file "$PROF_OUTPUT" \
    --csv \
    ./build/cnn_inference_gpu --model "$MODEL_DIR" --device gpu --batch_size "$BATCH_SIZE"

# Generate summary
echo "Generating summary..."
{
    echo "=== GPU Profiling Summary (Batch Size: $BATCH_SIZE) ==="
    echo ""
    echo "Full profile saved to: $PROF_OUTPUT"
    echo "CSV profile saved to: $CSV_OUTPUT"
    echo ""
    echo "=== Key Metrics ==="
    
    # Extract key metrics from nvprof output
    if [ -f "$PROF_OUTPUT" ]; then
        echo ""
        echo "GPU Trace (Top Operations):"
        grep -A 20 "GPU activities" "$PROF_OUTPUT" | head -30
        
        echo ""
        echo "API Calls:"
        grep -A 10 "API calls" "$PROF_OUTPUT" | head -20
        
        echo ""
        echo "Memory Operations:"
        grep -i "memcpy\|malloc\|free" "$PROF_OUTPUT" || echo "No memory operations found"
    fi
} > "$SUMMARY_OUTPUT"

echo ""
echo "=== Profiling Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo "Summary: $SUMMARY_OUTPUT"

