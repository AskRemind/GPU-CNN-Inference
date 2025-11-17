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

# Check if profiling tools are available (nsys for newer GPUs, nvprof for older)
if command -v nsys >/dev/null 2>&1; then
    PROFILER="nsys"
elif command -v nvprof >/dev/null 2>&1; then
    PROFILER="nvprof"
else
    echo "Error: Neither nsys nor nvprof found. Cannot run profiling."
    echo "Please install NVIDIA Nsight Systems (nsys) for profiling."
    exit 1
fi

echo "Using profiler: $PROFILER"

echo "=== GPU Profiling with nvprof ==="
echo "Batch size: $BATCH_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Output file names
PROF_OUTPUT="$OUTPUT_DIR/gpu_profiling_batch${BATCH_SIZE}.txt"
SUMMARY_OUTPUT="$OUTPUT_DIR/gpu_profiling_batch${BATCH_SIZE}_summary.txt"

# Run profiling based on available tool
echo "Running $PROFILER profiling..."

if [ "$PROFILER" = "nsys" ]; then
    # Use nsys for newer GPUs (compute capability 8.0+)
    NSYS_OUTPUT="$OUTPUT_DIR/gpu_profile_batch${BATCH_SIZE}"
    nsys profile \
        --output="$NSYS_OUTPUT" \
        --trace=cuda,nvtx \
        --force-overwrite=true \
        ./build/cnn_inference_gpu --model "$MODEL_DIR" --device gpu --batch_size "$BATCH_SIZE" 2>&1 | tee "$PROF_OUTPUT"
    
    # Convert nsys output to text summary
    if [ -f "${NSYS_OUTPUT}.nsys-rep" ]; then
        nsys stats --report gputrace "${NSYS_OUTPUT}.nsys-rep" >> "$PROF_OUTPUT" 2>&1
        echo "" >> "$PROF_OUTPUT"
        echo "=== API Trace ===" >> "$PROF_OUTPUT"
        nsys stats --report cudaapisum "${NSYS_OUTPUT}.nsys-rep" >> "$PROF_OUTPUT" 2>&1
    fi
else
    # Use nvprof for older GPUs
    nvprof \
        --print-gpu-trace \
        --print-api-trace \
        --log-file "$PROF_OUTPUT" \
        ./build/cnn_inference_gpu --model "$MODEL_DIR" --device gpu --batch_size "$BATCH_SIZE" 2>&1 | tee -a "$PROF_OUTPUT"
fi

# Generate summary
echo "Generating summary..."
{
    echo "=== GPU Profiling Summary (Batch Size: $BATCH_SIZE) ==="
    echo ""
    echo "Full profile saved to: $PROF_OUTPUT"
    echo ""
    echo "=== Key Metrics ==="
    
    # Extract key metrics from nvprof output
    if [ -f "$PROF_OUTPUT" ]; then
        echo ""
        echo "GPU Trace (Top Operations):"
        # Try different patterns for GPU activities
        if grep -q "GPU activities" "$PROF_OUTPUT"; then
            grep -A 20 "GPU activities" "$PROF_OUTPUT" | head -30
        elif grep -q "GPU time" "$PROF_OUTPUT"; then
            grep -A 20 "GPU time" "$PROF_OUTPUT" | head -30
        elif grep -q "CUDA Kernel" "$PROF_OUTPUT"; then
            grep -A 20 "CUDA Kernel" "$PROF_OUTPUT" | head -30
        else
            # Show first 50 lines if pattern not found
            echo "Showing first 50 lines of profile:"
            head -50 "$PROF_OUTPUT"
        fi
        
        echo ""
        echo "API Calls:"
        # Try different patterns for API calls
        if grep -q "API calls" "$PROF_OUTPUT"; then
            grep -A 10 "API calls" "$PROF_OUTPUT" | head -20
        elif grep -q "CUDA API" "$PROF_OUTPUT"; then
            grep -A 10 "CUDA API" "$PROF_OUTPUT" | head -20
        elif grep -q "cudaapisum" "$PROF_OUTPUT"; then
            grep -A 20 "cudaapisum" "$PROF_OUTPUT" | head -30
        else
            grep -i "cuda\|api" "$PROF_OUTPUT" | head -20 || echo "No API calls found in expected format"
        fi
        
        echo ""
        echo "Memory Operations:"
        grep -i "memcpy\|malloc\|free\|memory\|cudaMemcpy" "$PROF_OUTPUT" | head -20 || echo "No memory operations found"
    else
        echo "Error: Profiling output file not found: $PROF_OUTPUT"
    fi
} > "$SUMMARY_OUTPUT"

echo ""
echo "=== Profiling Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo "Summary: $SUMMARY_OUTPUT"

