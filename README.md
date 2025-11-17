# GPU-CNN-Inference

CUDA/C++ playground for reproducing VGG11 inference from scratch across three execution modes:

- **CPU (sequential)**: current baseline that runs the full forward pass on a single core.
- **CPU (OpenMP)**: planned multi-core speedup by parallelizing convolution, pooling, and GEMM loops.
- **GPU (CUDA)**: upcoming kernel suite with tiled convolution, pooling, GEMM, and lightweight element-wise layers.

The project targets the Food-101 dataset (scaled-down subset for development) and CIMS GPU machines. No vendor libraries other than CUDA runtime are used; all kernels and layers are handcrafted.

---

## Repository Layout

```
GPU-CNN-Inference/
├── src/                    # Source tree
│   ├── main.cpp            # CLI entry point / argument parsing
│   ├── common/             # Model loader, timers, shared utils
│   ├── cpu/                # Conv/FC/activation/pooling primitives + inference driver
│   └── layers/             # Layer base classes / interfaces
├── models/                 # VGG11 weights (binary .dat files, ignored in Git)
├── data/                   # Food-101 subset (images + meta splits, ignored in Git)
├── results/                # Benchmark outputs, plots, logs (ignored in Git)
├── check_server_environment.sh
├── split_dataset.py        # Helper to carve out Food-101 subsets
├── PROJECT_GUIDE.md        # Detailed milestone + experiment plan
├── SETUP.md                # Notes specific to local/CIMS setup
└── Makefile                # Build targets for cpu / multicore / gpu variants
```

> `data/` and `models/` are intentionally gitignored; keep them locally or on CIMS storage to avoid bloating the repository.

---

## Prerequisites

- GCC/Clang with C++17 support and OpenMP.
- CUDA toolkit ≥ 12.0 (for the GPU build path).
- CMake _not_ required; the root `Makefile` orchestrates each target.
- Food-101 subset extracted under `data/food-101` (class folders and `meta/` splits).
- Exported VGG11 weights placed under `models/` (`conv*.dat`, `fc*.dat`, `imagenet_classes.txt`).

Optional tooling:

- `nvprof` / `nsys` / `ncu` for GPU profiling.
- Python 3.10+ if you want to re-run `split_dataset.py`.

---

## Building

```bash
# CPU sequential (default target)
make cpu

# CPU OpenMP variant (work in progress)
make multicore

# CUDA kernels (work in progress)
make gpu
```

Executables are emitted inside `build/`. Use `make clean` to wipe intermediates.

---

## Running Inference

```bash
./build/cnn_inference_cpu --model models/ --device cpu --image data/food-101-subset/test/<class>/<image>.jpg
```

Current state:

- Image loader is under construction, so `main.cpp` feeds a dummy tensor. Replace `dummy_input` with preprocessed image data (CHW, float32, normalized) once the loader lands.
- GPU path exits early with a placeholder message until kernels are added.

Planned CLI flags (already parsed but not fully used):

- `--model DIR` — Directory containing `.dat` weight shards and class labels.
- `--image FILE` — Input RGB image (will trigger preprocessing once implemented).
- `--device cpu|multicore|gpu` — Backend selector.
- `--batch N` — Will arrive alongside batch inference API.

---

## Datasets & Weights

1. **Download Food-101**
   ```bash
   wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   tar -xzf food-101.tar.gz
   ```
2. (Optional) Use `split_dataset.py` to sample a 14-class subset for quicker turnaround:  
   `python split_dataset.py --source food-101 --dest data/food-101-subset --classes 14`
3. Place `data/food-101-subset/` (or full `food-101/`) inside the repo root.
4. Copy pre-exported VGG11 weights into `models/` keeping the `.dat` naming scheme expected by `model_loader.cpp`.

Both folders remain local-only; they are excluded from Git to keep the repo lightweight.

---

## Syncing to CIMS

Use NYU CIMS login (`netid@access.cims.nyu.edu`) and `scp`/`rsync` to transfer assets that are ignored in Git:

```bash
# One-time code push happens via git; send only large assets manually
scp -r models data/food-101-subset \
    netid@access.cims.nyu.edu:~/GPU-CNN-Inference/
```

On the remote machine:

```bash
module load cuda-12.6
cd GPU-CNN-Inference
make cpu            # or make multicore / make gpu
./build/cnn_inference_cpu --model models --device cpu
```

---

## Roadmap Snapshot

- [ ] Image pre-processing (resize, normalize, CHW pack).
- [ ] Accuracy validation vs. PyTorch baseline.
- [ ] OpenMP parallel loops for conv / FC / pool.
- [ ] CUDA kernels with shared-memory tiling and kernel fusion opportunities.
- [ ] Benchmark harness + plotting scripts to capture speedup, throughput, and memory usage.

Refer to `PROJECT_GUIDE.md` for the detailed milestone checklist and experiment slate.

# GPU-Accelerated CNN Inference

A C++/CUDA implementation of Convolutional Neural Network (CNN) inference for image classification. This project implements CNN inference from scratch, including convolution, pooling, activation functions, fully connected layers, and softmax.

## Overview

This repository contains:
- **C++/CUDA source code** for CNN inference implementation
- **Pre-trained VGG11 model weights** (exported to binary format)
- **Food-101 test dataset subset** (14 classes, ~14,000 images)

## Repository Contents

```
.
├── src/                    # Source code
│   ├── common/            # Shared utilities (model loader, timer)
│   ├── layers/             # Layer base classes
│   ├── cpu/                # CPU implementations
│   ├── gpu/                # GPU implementations (TODO)
│   └── main.cpp            # Main entry point
├── models/                 # Pre-trained VGG11 weights (~500MB)
│   ├── *.dat               # Binary weight files (22 files)
│   ├── model_info.json     # Model architecture information
│   └── imagenet_classes.txt
├── data/                   # Test dataset
│   └── food-101-subset/    # Food-101 subset (14 classes)
├── Makefile                # Build configuration
└── README.md
```

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

### 2. Verify Model Weights and Dataset

The repository includes pre-trained model weights and a test dataset subset:

```bash
# Verify model weights exist
ls models/*.dat

# Verify test dataset exists
ls data/food-101-subset/
```

### 3. Build Project

```bash
# Load CUDA module (if using module system)
module load cuda-12.6

# Build CPU sequential implementation
make cpu

# Or build all implementations
make all
```

### 4. Run Inference

```bash
# Run CPU inference
./build/cnn_inference_cpu --model models/ --device cpu

# Run with image (when image loader is implemented)
./build/cnn_inference_cpu --model models/ --image data/test.jpg --device cpu
```

## Model Information

- **Architecture**: VGG11 (pre-trained on ImageNet)
- **Input**: 224×224×3 RGB images
- **Output**: 1000 ImageNet class probabilities

### Model Layers

- **Convolutional Layers**: 8 conv layers (3→64→128→256→256→512→512→512→512 channels)
- **Pooling**: 5 MaxPool layers (2×2, stride 2)
- **Activation**: ReLU after each conv layer
- **Fully Connected**: 3 FC layers (25088→4096→4096→1000)
- **Output**: Softmax

## Dataset Information

The repository includes a subset of the Food-101 dataset:

- **14 food categories**: apple_pie, bread_pudding, cheesecake, chicken_curry, chocolate_cake, donuts, french_fries, hamburger, ice_cream, pizza, sushi, tacos, tiramisu, waffles
- **Training images**: ~10,500 (750 per class)
- **Test images**: ~3,500 (250 per class)

Full Food-101 dataset (101 classes, ~101,000 images) can be downloaded from:
https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

## Implementation Status

- [x] Model weight loader (C++)
- [x] CPU sequential implementation
- [x] Layer implementations: Conv2D, ReLU, MaxPool2D, Linear, Softmax
- [ ] CPU multicore implementation (OpenMP)
- [ ] GPU CUDA implementation
- [ ] Image loader
- [ ] Benchmark scripts

## Build Targets

- `make cpu` - Build CPU sequential implementation
- `make multicore` - Build CPU multicore implementation (OpenMP)
- `make gpu` - Build GPU CUDA implementation
- `make all` - Build all implementations
- `make clean` - Clean build artifacts

## Project Structure

### Source Code Organization

- **`src/common/`**: Shared utilities
  - `model_loader.h/cpp`: Loads pre-trained weights from binary files
  - `timer.h`: Performance timing utilities

- **`src/layers/`**: Layer base classes
  - `layer_base.h`: Abstract base class for all layers

- **`src/cpu/`**: CPU implementations
  - `cpu_conv2d.h/cpp`: 2D convolution
  - `cpu_layers.h/cpp`: ReLU, MaxPool, Linear, Softmax
  - `cpu_inference.h/cpp`: CPU inference engine

- **`src/gpu/`**: GPU CUDA implementations (TODO)

## Notes

- Model weights are pre-exported binary files - no PyTorch or Python required
- The test dataset subset is included for quick testing
- All inference code is implemented from scratch (no cuDNN or other proprietary libraries)

## License

This project is for educational purposes.

Model weights are from PyTorch torchvision (ImageNet pre-trained VGG11), which are publicly available.
