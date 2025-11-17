# GPU-Accelerated CNN Inference

A C++/CUDA implementation of Convolutional Neural Network (CNN) inference for image classification. This project implements VGG11 inference from scratch using three different approaches: CPU Sequential, CPU Multicore (OpenMP), and GPU CUDA.

## Overview

This project provides three implementations of VGG11 inference for performance comparison:

- **CPU Sequential**: Baseline single-threaded implementation
- **CPU Multicore**: Multi-core parallelized implementation using OpenMP
- **GPU CUDA**: GPU-accelerated implementation using CUDA kernels

All implementations are written from scratch without using proprietary libraries like cuDNN.

## Model Architecture

- **Model**: VGG11 (pre-trained on ImageNet)
- **Input**: 224×224×3 RGB images
- **Output**: 1000 ImageNet class probabilities

### Layer Architecture

- **8 Convolutional Layers**: 3→64→128→256→256→512→512→512→512 channels
- **5 MaxPool Layers**: 2×2 max pooling with stride 2
- **ReLU Activation**: After each convolutional layer
- **3 Fully Connected Layers**: 25088→4096→4096→1000
- **Softmax Output**: Probability distribution over 1000 classes

## Prerequisites

- **CUDA Toolkit** (>= 11.0)
- **C++17 compiler** (g++)
- **Make**
- **OpenMP** (for multicore CPU implementation)
- **Python 3** with **PyTorch** and **torchvision** (for weight export)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/AskRemind/GPU-CNN-Inference.git
cd GPU-CNN-Inference
```

### 2. Prepare Model Weights

**Option A: Export weights from PyTorch (Recommended)**

If you have PyTorch installed, you can export weights from a pre-trained VGG11 model:

```bash
# Install PyTorch (if not already installed)
pip install torch torchvision

# Export weights
python3 export_weights.py --output_dir models/
```

This will generate all weight files (`.dat`), `model_info.json`, and `imagenet_classes.txt` in the `models/` directory.

**Option B: Use pre-exported weights**

If you already have the weight files, place them in the `models/` directory:

```
models/
├── conv0.weight.dat
├── conv0.bias.dat
├── ...
├── fc2.weight.dat
├── fc2.bias.dat
├── model_info.json
└── imagenet_classes.txt
```

**Note**: Weight files are excluded from the repository due to size (~500MB). You need to either export them yourself or obtain them separately.

### 3. Build Project

```bash
# Build all implementations
make all

# Or build individually
make cpu          # CPU Sequential
make cpu_multicore  # CPU Multicore (OpenMP)
make gpu          # GPU CUDA
```

### 4. Run Inference

```bash
# CPU Sequential
./build/cnn_inference_cpu --model models/ --device cpu --batch_size 1

# CPU Multicore
./build/cnn_inference_cpu_multicore --model models/ --device cpu_multicore --batch_size 1

# GPU CUDA
./build/cnn_inference_gpu --model models/ --device gpu --batch_size 1
```

## Reproducing Experiments

### Experiment 1: Baseline Performance (batch_size=1)

Test each implementation with batch_size=1:

```bash
# CPU Sequential
./benchmark_cpu.sh --runs 20 --batch 1 --output results/baseline/cpu_batch1.csv

# CPU Multicore
./benchmark_cpu_multicore.sh --runs 20 --batch 1 --output results/baseline/cpu_multicore_batch1.csv

# GPU CUDA
./benchmark_gpu.sh --runs 20 --batch 1 --output results/baseline/gpu_batch1.csv
```

### Experiment 2: Scalability Testing (Different Batch Sizes)

Run scalability tests across different batch sizes:

```bash
# This will test batch_size = [1, 2, 4, 8, 16, 32] for all three implementations
./benchmark_scalability.sh --runs 10 --batch_sizes "1,2,4,8,16,32" --output_dir results/scalability
```

Individual batch size tests:

```bash
# CPU Sequential
for bs in 1 2 4 8 16 32; do
    ./benchmark_cpu.sh --runs 10 --batch $bs --output results/scalability/cpu_batch${bs}.csv
done

# CPU Multicore
for bs in 1 2 4 8 16 32; do
    ./benchmark_cpu_multicore.sh --runs 10 --batch $bs --output results/scalability/cpu_multicore_batch${bs}.csv
done

# GPU CUDA
for bs in 1 2 4 8 16 32; do
    ./benchmark_gpu.sh --runs 10 --batch $bs --output results/scalability/gpu_batch${bs}.csv
done
```

### Experiment 3: Hardware Scalability (Different GPUs)

Test on different GPU nodes:

```bash
# On GPU node 1 (e.g., cuda5)
./benchmark_gpu.sh --runs 10 --batch 16 --output results/hardware/cuda5_batch16.csv

# On GPU node 2 (e.g., cuda4)
./benchmark_gpu.sh --runs 10 --batch 16 --output results/hardware/cuda4_batch16.csv
```

### Experiment 4: GPU Profiling

Profile GPU performance to analyze kernel execution and memory usage:

```bash
# Run profiling (automatically uses nsys for newer GPUs, nvprof for older GPUs)
./benchmark_gpu_profiling.sh --batch_size 16 --output_dir results/profiling
```

This generates:
- `gpu_profiling_batch16.txt` - Full profiling output
- `gpu_profiling_batch16_summary.txt` - Summary report
- `gpu_profile_batch16.nsys-rep` - Nsight Systems report (for visual analysis)

To generate text reports from the .nsys-rep file:

```bash
nsys stats --report gputrace results/profiling/gpu_profile_batch16.nsys-rep > results/profiling/gpu_trace_batch16.txt
nsys stats --report cudaapisum results/profiling/gpu_profile_batch16.nsys-rep > results/profiling/gpu_api_batch16.txt
```

## Benchmark Scripts

The repository includes the following benchmark scripts:

1. **benchmark_cpu.sh** - Benchmark CPU Sequential implementation
2. **benchmark_cpu_multicore.sh** - Benchmark CPU Multicore implementation
3. **benchmark_gpu.sh** - Benchmark GPU CUDA implementation
4. **benchmark_scalability.sh** - Run scalability tests across multiple batch sizes
5. **benchmark_gpu_profiling.sh** - Profile GPU performance with nsys/nvprof

### Script Usage

All benchmark scripts support similar arguments:

```bash
# Basic usage
./benchmark_cpu.sh --runs 10 --batch 8 --output results/cpu_batch8.csv

# With model directory
./benchmark_cpu.sh --runs 10 --batch 8 --model models/ --output results/cpu_batch8.csv

# Scalability test (all batch sizes)
./benchmark_scalability.sh --runs 10 --batch_sizes "1,2,4,8,16,32" --output_dir results/scalability
```

## Project Structure

```
.
├── src/                              # Source code
│   ├── common/                       # Shared utilities
│   │   ├── model_loader.h/cpp       # Weight file loader
│   │   └── timer.h                  # Performance timer
│   ├── layers/                       # Layer base classes
│   │   └── layer_base.h             # Abstract layer interface
│   ├── cpu/                          # CPU Sequential implementation
│   │   ├── cpu_conv2d.h/cpp
│   │   ├── cpu_layers.h/cpp
│   │   └── cpu_inference.h/cpp
│   ├── cpu_multicore/                # CPU Multicore implementation
│   │   ├── cpu_multicore_conv2d.h/cpp
│   │   ├── cpu_multicore_layers.h/cpp
│   │   └── cpu_multicore_inference.h/cpp
│   ├── gpu/                          # GPU CUDA implementation
│   │   ├── gpu_conv2d.h/cu
│   │   ├── gpu_layers.h/cu
│   │   └── gpu_inference.h/cu
│   └── main.cpp                      # Main entry point
├── models/                           # Model weights (not in repo)
│   ├── *.dat                         # Binary weight files (22 files)
│   ├── model_info.json              # Model architecture info
│   └── imagenet_classes.txt         # Class names
├── results/                          # Experiment results (not in repo)
│   ├── scalability/                 # Scalability test results
│   ├── hardware/                    # Hardware scalability results
│   └── profiling/                   # GPU profiling results
├── export_weights.py                # Python script to export weights from PyTorch
├── benchmark_*.sh                   # Benchmark scripts
├── Makefile                         # Build configuration
└── README.md                        # This file
```

## Build Targets

| Target | Description |
|--------|-------------|
| `make cpu` | Build CPU Sequential implementation |
| `make cpu_multicore` | Build CPU Multicore implementation |
| `make gpu` | Build GPU CUDA implementation |
| `make all` | Build all implementations |
| `make clean` | Clean build artifacts |

## Command Line Arguments

```
Usage: ./cnn_inference [options]

Options:
  --model DIR       Model directory containing weight files (default: models/)
  --device DEV      Device: cpu, cpu_multicore, or gpu (default: cpu)
  --batch_size N    Batch size for inference (default: 1)
  --help            Show help message
```

## Model Weights

### Weight Format

Weights are stored as binary `.dat` files containing float32 arrays in C++ array order (row-major).

Each layer has two files:
- `{layer_name}.weight.dat` - Weight matrix/kernel
- `{layer_name}.bias.dat` - Bias vector

### Weight Export Script

The `export_weights.py` script extracts weights from a PyTorch pre-trained VGG11 model:

```bash
python3 export_weights.py --output_dir models/
```

**Requirements**:
- Python 3
- PyTorch (>= 1.8)
- torchvision

**Generated Files**:
- 22 weight files (`.dat` format)
- `model_info.json` - Model architecture information
- `imagenet_classes.txt` - ImageNet class names (placeholder, download full list separately)

### Weight File Structure

- **Convolutional layers**: `conv0` through `conv7`
  - Weight: `[out_channels, in_channels, kernel_h, kernel_w]`
  - Bias: `[out_channels]`

- **Fully connected layers**: `fc0`, `fc1`, `fc2`
  - Weight: `[out_features, in_features]`
  - Bias: `[out_features]`

## Performance Metrics

The experiments measure:

- **Speedup**: GPU vs CPU implementations (relative to CPU Sequential baseline)
- **Scalability**: Performance with varying batch sizes (1, 2, 4, 8, 16, 32)
- **Memory Usage**: Peak memory consumption (CPU memory and GPU memory)
- **Throughput**: Images processed per second

## Implementation Details

### CPU Sequential
- Single-threaded baseline implementation
- Direct convolution loops
- Optimized with compiler flags (`-O3`, `-march=native`)
- Ping-pong buffers to avoid unnecessary memory copies

### CPU Multicore
- OpenMP parallelization of convolution, pooling, and matrix multiplication
- Parallel loops over output feature maps and channels
- Removed batch-level parallelization to avoid overhead for small batches
- Ping-pong buffers for memory efficiency

### GPU CUDA
- Custom CUDA kernels for all operations
- GPU memory pool with ping-pong buffers
- Minimal Host-GPU data transfers (only input and output)
- All intermediate data stays on GPU
- Optimized kernel launch configurations

## Results

Results are saved in CSV format in the `results/` directory:

- `results/scalability/*.csv` - Scalability test results
- `results/hardware/*.csv` - Hardware scalability results
- `results/profiling/*.txt` - GPU profiling reports

Each CSV file contains:
- `run` - Run number
- `batch_size` - Batch size used
- `time_ms` - Inference time in milliseconds
- `memory_mb` - Peak CPU memory usage in MB
- `gpu_memory_mb` - Peak GPU memory usage in MB (GPU tests only)

## License

This project is for educational purposes.

Model weights are from PyTorch torchvision (ImageNet pre-trained VGG11), which are publicly available under the PyTorch license.

## Contributing

This is an academic project. Please refer to the course guidelines for submission requirements.

## Acknowledgments

- VGG11 architecture and pre-trained weights from PyTorch torchvision
- ImageNet for pre-training data
- NVIDIA CUDA Toolkit and documentation
