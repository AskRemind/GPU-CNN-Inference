#!/bin/bash
# Script to check available libraries and tools on CIMS CUDA cluster

echo "=========================================="
echo "CIMS Server Environment Check"
echo "=========================================="
echo

echo "=== System Information ==="
uname -a
echo

echo "=== CUDA Information ==="
if command -v nvcc &> /dev/null; then
    echo "CUDA Compiler (nvcc):"
    nvcc --version
    echo
    echo "CUDA Toolkit location:"
    which nvcc
else
    echo "nvcc not found in PATH"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
else
    echo "nvidia-smi not found"
fi
echo

echo "=== Compiler Information ==="
if command -v gcc &> /dev/null; then
    echo "GCC version:"
    gcc --version | head -n 1
fi

if command -v g++ &> /dev/null; then
    echo "G++ version:"
    g++ --version | head -n 1
fi
echo

echo "=== CMake Information ==="
if command -v cmake &> /dev/null; then
    cmake --version
else
    echo "cmake not found"
fi
echo

echo "=== OpenMP Information ==="
if [ -f /usr/lib/libomp.so ] || [ -f /usr/lib64/libomp.so ]; then
    echo "OpenMP libraries found:"
    find /usr/lib* -name "libomp.so*" 2>/dev/null
else
    echo "OpenMP libraries not found in standard locations"
fi
echo

echo "=== BLAS/LAPACK Libraries ==="
echo "Checking for BLAS libraries..."
if [ -f /usr/lib/libblas.so ] || [ -f /usr/lib64/libblas.so ]; then
    echo "✓ libblas found"
fi
if [ -f /usr/lib/libopenblas.so ] || [ -f /usr/lib64/libopenblas.so ]; then
    echo "✓ libopenblas found"
fi
if [ -f /usr/lib/libatlas.so ] || [ -f /usr/lib64/libatlas.so ]; then
    echo "✓ libatlas found"
fi
echo

echo "=== Library Search Paths ==="
echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH
echo
echo "Standard library paths:"
echo "/usr/lib:"
ls -1 /usr/lib/*.so* 2>/dev/null | head -20
echo
echo "/usr/lib64:"
ls -1 /usr/lib64/*.so* 2>/dev/null | head -20
echo

echo "=== Module System (if available) ==="
if command -v module &> /dev/null; then
    echo "Available modules:"
    module avail 2>&1 | head -50
else
    echo "module command not available"
fi
echo

echo "=== Python Environment ==="
if command -v python3 &> /dev/null; then
    echo "Python version:"
    python3 --version
    echo
    echo "Python packages:"
    python3 -c "import sys; print('Python path:', sys.executable)"
    echo
    echo "Checking common packages:"
    python3 -c "import numpy; print('✓ numpy')" 2>/dev/null || echo "✗ numpy not found"
    python3 -c "import torch; print('✓ torch')" 2>/dev/null || echo "✗ torch not found"
    python3 -c "import cv2; print('✓ opencv')" 2>/dev/null || echo "✗ opencv not found"
    python3 -c "import PIL; print('✓ Pillow')" 2>/dev/null || echo "✗ Pillow not found"
fi
echo

echo "=== Checking CUDA Libraries ==="
if [ -n "$CUDA_HOME" ] || [ -n "$CUDA_PATH" ]; then
    CUDA_DIR=${CUDA_HOME:-$CUDA_PATH}
    echo "CUDA directory: $CUDA_DIR"
    if [ -d "$CUDA_DIR/lib64" ]; then
        echo "CUDA libraries:"
        ls -1 "$CUDA_DIR/lib64"/*.so* 2>/dev/null | head -20
    fi
else
    echo "CUDA_HOME not set, checking common locations:"
    for dir in /usr/local/cuda* /opt/cuda*; do
        if [ -d "$dir/lib64" ]; then
            echo "Found: $dir"
            ls -1 "$dir/lib64"/*.so* 2>/dev/null | head -10
        fi
    done
fi
echo

echo "=========================================="
echo "Check complete!"
echo "=========================================="


