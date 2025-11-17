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
    
    # Measure execution time and memory using /usr/bin/time -v
    if command -v /usr/bin/time >/dev/null 2>&1; then
        # Use /usr/bin/time -v for detailed statistics
        TIME_OUTPUT=$(/usr/bin/time -v ./build/cnn_inference_cpu --model "$MODEL_DIR" --device cpu --batch_size "$BATCH" 2>&1)
        
        # Extract time from program output (if available)
        # Match "Inference completed in X.XX ms" - extract the number before " ms"
        TIME_MS=$(echo "$TIME_OUTPUT" | grep "Inference completed" | sed -n 's/.*Inference completed in \([0-9]\+\.[0-9]\+\) ms.*/\1/p' | head -1)
        
        # If time extraction failed, use elapsed time from /usr/bin/time
        if [ -z "$TIME_MS" ]; then
            ELAPSED=$(echo "$TIME_OUTPUT" | grep "Elapsed (wall clock) time" | grep -oE '[0-9]+\.[0-9]+' | head -1)
            if [ -n "$ELAPSED" ]; then
                TIME_MS=$(awk "BEGIN {print $ELAPSED * 1000}")
            else
                TIME_MS="N/A"
            fi
        fi
        
        # Extract peak memory usage (Maximum resident set size) in KB, convert to MB
        MEMORY_KB=$(echo "$TIME_OUTPUT" | grep "Maximum resident set size (kbytes)" | grep -oE '[0-9]+' | head -1)
        if [ -n "$MEMORY_KB" ] && [ "$MEMORY_KB" != "0" ]; then
            MEMORY_MB=$(awk "BEGIN {print $MEMORY_KB / 1024}")
        else
            MEMORY_MB="N/A"
        fi
    else
        # Fallback: Use date and ps (less accurate)
        START=$(date +%s.%N 2>/dev/null || date +%s)
        
        # Run in background and monitor memory
        ./build/cnn_inference_cpu --model "$MODEL_DIR" --device cpu --batch_size "$BATCH" > /tmp/inference_output_$$.txt 2>&1 &
        INFERENCE_PID=$!
        
        # Monitor memory while process is running
        PEAK_MEMORY_KB=0
        while kill -0 $INFERENCE_PID 2>/dev/null; do
            CURRENT_MEM=$(ps -o rss= -p $INFERENCE_PID 2>/dev/null | awk '{print $1}')
            if [ -n "$CURRENT_MEM" ] && [ "$CURRENT_MEM" -gt "$PEAK_MEMORY_KB" ]; then
                PEAK_MEMORY_KB=$CURRENT_MEM
            fi
            sleep 0.1
        done
        
        wait $INFERENCE_PID
        END=$(date +%s.%N 2>/dev/null || date +%s)
        
        # Read output
        OUTPUT_TEXT=$(cat /tmp/inference_output_$$.txt)
        rm -f /tmp/inference_output_$$.txt
        
        # Extract time from output
        # Match "Inference completed in X.XX ms" - extract the number before " ms"
        TIME_MS=$(echo "$OUTPUT_TEXT" | grep "Inference completed" | sed -n 's/.*Inference completed in \([0-9]\+\.[0-9]\+\) ms.*/\1/p' | head -1)
        
        # If time extraction failed, use wall-clock time
        if [ -z "$TIME_MS" ]; then
            ELAPSED=$(awk "BEGIN {print $END - $START}")
            TIME_MS=$(awk "BEGIN {print $ELAPSED * 1000}")
        fi
        
        # Convert memory from KB to MB
        if [ "$PEAK_MEMORY_KB" -gt 0 ]; then
            MEMORY_MB=$(awk "BEGIN {print $PEAK_MEMORY_KB / 1024}")
        else
            MEMORY_MB="N/A"
        fi
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

