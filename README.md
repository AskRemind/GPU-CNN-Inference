# GPU-Accelerated CNN Inference

A C++/CUDA implementation of Convolutional Neural Network (CNN) inference for image classification. This project implements CNN inference from scratch, including convolution, pooling, activation functions, fully connected layers, and softmax.

## Overview

This project provides three implementations of VGG11 inference:

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

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/AskRemind/GPU-CNN-Inference.git
cd GPU-CNN-Inference
```

### 2. Prepare Model Weights and Dataset

**Model Weights** (required):
- Place pre-trained VGG11 weights in `models/` directory
- Weights should be in binary `.dat` format (22 files)
- Files: `conv*.dat`, `fc*.dat`, `model_info.json`, `imagenet_classes.txt`

**Dataset** (optional for basic testing):
- Place test images in `data/` directory
- For Food-101 dataset: extract to `data/food-101-subset/`
- Dataset not included in repository due to size

### 3. Build Project

```bash
# Build CPU sequential implementation
make cpu

# Build CPU multicore implementation (OpenMP)
make multicore

# Build GPU CUDA implementation
make gpu

# Build all implementations
make all
```

### 4. Run Inference

```bash
# CPU sequential
./build/cnn_inference_cpu --model models/ --device cpu

# CPU multicore
./build/cnn_inference_multicore --model models/ --device cpu

# GPU CUDA
./build/cnn_inference_gpu --model models/ --device gpu

# With image input (when image loader is implemented)
./build/cnn_inference_cpu --model models/ --image data/test.jpg --device cpu
```

## Project Structure

```
.
├── src/                        # Source code
│   ├── common/                 # Shared utilities
│   │   ├── model_loader.h/cpp  # Weight file loader
│   │   └── timer.h             # Performance timer
│   ├── layers/                 # Layer base classes
│   │   └── layer_base.h        # Abstract layer interface
│   ├── cpu/                    # CPU implementations
│   │   ├── cpu_conv2d.h/cpp    # 2D convolution
│   │   ├── cpu_layers.h/cpp    # ReLU, MaxPool, Linear, Softmax
│   │   └── cpu_inference.h/cpp # CPU inference engine
│   ├── gpu/                    # GPU implementations (TODO)
│   └── main.cpp                # Main entry point
├── models/                     # Model weights (not in repo)
│   ├── *.dat                   # Binary weight files (22 files)
│   ├── model_info.json         # Model architecture info
│   └── imagenet_classes.txt    # Class names
├── data/                       # Dataset (not in repo)
│   └── food-101-subset/        # Food-101 subset (14 classes)
├── build/                      # Build output (generated)
├── Makefile                    # Build configuration
└── README.md
```

## Build Targets

| Target | Description |
|--------|-------------|
| `make cpu` | Build CPU sequential implementation |
| `make multicore` | Build CPU multicore implementation (OpenMP) |
| `make gpu` | Build GPU CUDA implementation |
| `make all` | Build all implementations |
| `make clean` | Clean build artifacts |

## Command Line Arguments

```
Usage: ./cnn_inference [options]

Options:
  --model DIR    Model directory containing weight files (default: models/)
  --image FILE   Input image file (JPEG/PNG)
  --device DEV   Device: cpu, multicore, or gpu (default: cpu)
  --batch N      Batch size for inference (default: 1)
  --help         Show help message
```

## Dataset Information

The project is designed for the Food-101 dataset:

- **Food-101**: 101 food categories, ~101,000 images
- **Subset**: 14 categories, ~14,000 images (for quick testing)
- **Source**: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

Download and extract to `data/` directory:

```bash
wget https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/food-101.tar.gz
tar -xzf food-101.tar.gz -C data/
```

## Implementation Details

### CPU Sequential
- Single-threaded baseline implementation
- Direct convolution loops
- Standard C++ implementation with compiler optimizations

### CPU Multicore
- OpenMP parallelization of convolution, pooling, and matrix multiplication
- Parallel loops over output feature maps and channels
- Thread-safe operations

### GPU CUDA
- Custom CUDA kernels for all operations
- Shared memory optimization for convolution
- Coalesced memory access patterns
- Kernel fusion opportunities (e.g., conv + ReLU)

## Performance Evaluation

The project measures:

- **Speedup**: GPU vs CPU implementations
- **Scalability**: Performance with varying batch sizes and image resolutions
- **Memory Usage**: Peak memory consumption for each implementation
- **Throughput**: Images processed per second

Benchmark scripts and results analysis tools will be provided in `results/` directory.

## Implementation Status

- [x] Model weight loader (C++)
- [x] CPU sequential implementation
- [x] Layer implementations: Conv2D, ReLU, MaxPool2D, Linear, Softmax
- [ ] CPU multicore implementation (OpenMP)
- [ ] GPU CUDA implementation
- [ ] Image loader (JPEG/PNG)
- [ ] Benchmark scripts
- [ ] Performance profiling tools

## License

This project is for educational purposes.

Model weights are from PyTorch torchvision (ImageNet pre-trained VGG11), which are publicly available.

## Contributing

This is an academic project. Please refer to the course guidelines for submission requirements.

## Acknowledgments

- VGG11 architecture and pre-trained weights from PyTorch torchvision
- Food-101 dataset from ETH Zurich
- ImageNet for pre-training data
