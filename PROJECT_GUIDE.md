# Complete Project Guide: GPU-Accelerated CNN Inference

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Requirements](#project-requirements)
3. [Implementation Plan](#implementation-plan)
4. [Experimental Procedure](#experimental-procedure)
5. [Performance Analysis](#performance-analysis)
6. [Results and Evaluation](#results-and-evaluation)
7. [Submission Requirements](#submission-requirements)

---

## 1. Project Overview

### 1.1 Objective

Implement CNN inference from scratch in C++/CUDA and compare performance across three implementations:
- **Sequential CPU** (baseline)
- **Multicore CPU** (OpenMP or pthreads)
- **GPU** (CUDA kernels)

### 1.2 Model Architecture

- **Model**: VGG11 (pre-trained on ImageNet)
- **Input**: 224×224×3 RGB images
- **Output**: 1000 ImageNet class probabilities
- **Architecture**:
  - 8 Convolutional layers (3→64→128→256→256→512→512→512→512 channels)
  - 5 MaxPool layers (2×2, stride 2)
  - ReLU activation after each conv layer
  - 3 Fully Connected layers (25088→4096→4096→1000)
  - Softmax output

### 1.3 Dataset

- **Primary**: Food-101 subset (14 classes, ~14,000 images)
- **Full dataset**: Food-101 (101 classes, ~101,000 images)
- **Source**: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

### 1.4 Key Constraints

- ✅ Implement from scratch (no cuDNN or proprietary libraries)
- ✅ Must work on CIMS machines
- ✅ Cannot use libraries not available on CIMS (e.g., PyTorch at runtime)
- ✅ Pre-trained weights are exported to binary format

---

## 2. Project Requirements

### 2.1 Implementation Requirements

#### 2.1.1 Core Operations (Must Implement from Scratch)

1. **Convolution (Conv2D)**
   - Direct convolution (no FFT or Winograd)
   - Support for padding, stride, dilation
   - Handle variable kernel sizes and channel depths

2. **Pooling (MaxPool2D)**
   - 2×2 max pooling with stride 2
   - Efficient memory access patterns

3. **Activation (ReLU)**
   - Element-wise ReLU operation
   - In-place operations for memory efficiency

4. **Fully Connected (Linear)**
   - Matrix multiplication (GEMM)
   - Support for large weight matrices

5. **Softmax**
   - Numerical stability (subtract max before exp)
   - Normalize to probability distribution

#### 2.1.2 Three Implementations

1. **CPU Sequential**
   - Single-threaded baseline
   - Standard C++ implementation
   - Optimize with compiler flags (-O3, -march=native)

2. **CPU Multicore**
   - OpenMP or pthreads parallelization
   - Parallelize convolution loops, matrix multiplication
   - Measure speedup vs. sequential

3. **GPU CUDA**
   - CUDA kernels for all operations
   - Optimize memory access (coalesced, shared memory)
   - Optimize computation (thread blocks, grid dimensions)

### 2.2 Performance Metrics

#### 2.2.1 Speedup Analysis

- **Sequential CPU Baseline**: Measure inference time for single image
- **Multicore CPU Speedup**: `Speedup = T_sequential / T_multicore`
- **GPU Speedup**: `Speedup = T_sequential / T_gpu`

Target metrics:
- Multicore CPU: 4-8× speedup (depending on core count)
- GPU: 10-100× speedup (depending on GPU)

#### 2.2.2 Problem Scalability

Test with varying:
- **Batch sizes**: 1, 4, 8, 16, 32, 64
- **Image resolutions**: 224×224, 256×256, 512×512

Measure:
- Time per image vs. batch size
- Throughput (images/second) vs. batch size
- Memory usage vs. batch size/resolution

#### 2.2.3 Hardware Scalability

Test on different GPUs (if available):
- Compare performance on different GPU generations
- Analyze compute capability impact
- Memory bandwidth analysis

#### 2.2.4 Memory Usage

- Peak memory consumption for each implementation
- GPU memory usage (device memory)
- CPU memory usage (host memory)

### 2.3 Correctness Verification

- Compare outputs across implementations (CPU sequential, multicore, GPU)
- Verify against reference implementation (PyTorch) for single image
- Numerical precision analysis (float32 differences)

---

## 3. Implementation Plan

### 3.1 Phase 1: Foundation (Completed)

- [x] Project structure setup
- [x] Model weight loader (binary format)
- [x] CPU sequential implementation
- [x] Basic layer implementations (Conv2D, ReLU, MaxPool, Linear, Softmax)

### 3.2 Phase 2: CPU Multicore (TODO)

- [ ] Identify parallelizable loops
  - Convolution: output feature maps, input channels
  - Matrix multiplication: rows, columns
  - Pooling: output regions
- [ ] Implement OpenMP parallelization
  - `#pragma omp parallel for` with proper scheduling
  - Thread affinity tuning
- [ ] Optimize memory access patterns
- [ ] Benchmark and tune thread count

### 3.3 Phase 3: GPU CUDA (TODO)

- [ ] Convolution CUDA kernel
  - Shared memory for input tiles
  - Shared memory for weight tiles
  - Coalesced global memory access
  - Optimization: kernel fusion (conv + ReLU)
  
- [ ] Pooling CUDA kernel
  - Shared memory reduction
  - Efficient thread block organization
  
- [ ] Matrix Multiplication (GEMM)
  - Shared memory tiling
  - Register blocking
  - Multiple kernel variants for different sizes
  
- [ ] ReLU and Softmax kernels
  - Simple element-wise operations
  - Consider kernel fusion opportunities
  
- [ ] Memory management
  - Pre-allocate device memory
  - Minimize host-device transfers
  - Memory pooling for intermediate results

### 3.4 Phase 4: Image Loading (TODO)

- [ ] JPEG/PNG image loader (stb_image or similar)
- [ ] Image preprocessing
  - Resize to 224×224
  - Normalize pixel values
  - RGB to float32 conversion

### 3.5 Phase 5: Benchmarking (TODO)

- [ ] Benchmark script
  - Multiple runs for statistical significance
  - Warm-up runs
  - Time measurement (wall-clock and CPU time)
- [ ] Data collection
  - CSV output for batch sizes, resolutions
  - Memory profiling (nvprof, nvidia-smi)
- [ ] Visualization
  - Speedup plots
  - Throughput plots
  - Memory usage plots

---

## 4. Experimental Procedure

### 4.1 Setup

#### 4.1.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/AskRemind/GPU-CNN-Inference.git
cd GPU-CNN-Inference

# Load CUDA module (on CIMS)
module load cuda-12.6

# Verify CUDA
nvcc --version
nvidia-smi
```

#### 4.1.2 Data Preparation

```bash
# Upload model weights (if not in repo)
scp -r models/ server:~/GPU-CNN-Inference/

# Upload test dataset (if needed)
scp -r data/food-101-subset/ server:~/GPU-CNN-Inference/data/
```

#### 4.1.3 Build

```bash
# Build all implementations
make all

# Or build individually
make cpu        # Sequential CPU
make multicore  # Multicore CPU (OpenMP)
make gpu        # GPU CUDA
```

### 4.2 Experiment 1: Baseline Performance

**Objective**: Establish CPU sequential baseline

```bash
# Run CPU sequential inference
./build/cnn_inference_cpu --model models/ --device cpu --batch 1

# Run multiple times for average
for i in {1..10}; do
    ./build/cnn_inference_cpu --model models/ --device cpu --batch 1
done | grep "Inference completed" | awk '{sum+=$4; n++} END {print sum/n " ms"}'
```

**Measure**:
- Average inference time
- Standard deviation
- Memory usage (htop or /proc/self/status)

### 4.3 Experiment 2: Batch Size Scalability

**Objective**: Measure performance vs. batch size

```bash
# Test different batch sizes
for batch in 1 4 8 16 32 64; do
    echo "Batch size: $batch"
    ./build/cnn_inference_cpu --model models/ --device cpu --batch $batch
    ./build/cnn_inference_multicore --model models/ --device cpu --batch $batch
    ./build/cnn_inference_gpu --model models/ --device gpu --batch $batch
done
```

**Measure**:
- Time per image vs. batch size
- Throughput (images/sec) vs. batch size
- Memory usage vs. batch size

**Analysis**:
- Identify optimal batch size
- Explain throughput saturation
- Compare CPU vs. GPU scaling

### 4.4 Experiment 3: Resolution Scalability

**Objective**: Measure performance vs. image resolution

```bash
# Test different resolutions (requires implementation)
for res in 224 256 512; do
    echo "Resolution: ${res}x${res}"
    ./build/cnn_inference_gpu --model models/ --device gpu --resolution $res
done
```

**Measure**:
- Inference time vs. resolution
- Memory usage vs. resolution
- Compute intensity vs. resolution

### 4.5 Experiment 4: Multicore CPU Speedup

**Objective**: Measure OpenMP speedup

```bash
# Test different thread counts
for threads in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$threads
    echo "Threads: $threads"
    ./build/cnn_inference_multicore --model models/ --device cpu --batch 8
done
```

**Measure**:
- Speedup vs. thread count
- Efficiency (speedup/threads)
- Identify optimal thread count

**Analysis**:
- Compare to theoretical speedup (Amdahl's law)
- Identify bottlenecks
- Explain efficiency drop

### 4.6 Experiment 5: GPU Performance

**Objective**: Measure GPU speedup and optimization impact

```bash
# Baseline GPU
./build/cnn_inference_gpu --model models/ --device gpu --batch 1

# Optimized versions (if multiple kernels)
./build/cnn_inference_gpu --model models/ --device gpu --batch 1 --kernel optimized

# Profile GPU kernels
nvprof ./build/cnn_inference_gpu --model models/ --device gpu --batch 16
```

**Measure**:
- GPU speedup vs. CPU sequential
- GPU speedup vs. CPU multicore
- Kernel execution time breakdown
- Memory bandwidth utilization
- GPU utilization (nvidia-smi)

**Profiling**:
```bash
# Detailed profiling
nvprof --print-gpu-trace ./build/cnn_inference_gpu --model models/ --device gpu

# Memory analysis
nvprof --print-api-trace ./build/cnn_inference_gpu --model models/ --device gpu

# Kernel analysis
nvprof --kernels "conv2d*" --print-gpu-summary ./build/cnn_inference_gpu
```

### 4.7 Experiment 6: Hardware Scalability (if multiple GPUs available)

**Objective**: Compare performance across GPU generations

```bash
# Test on different GPUs
CUDA_VISIBLE_DEVICES=0 ./build/cnn_inference_gpu --model models/ --device gpu
CUDA_VISIBLE_DEVICES=1 ./build/cnn_inference_gpu --model models/ --device gpu
```

**Measure**:
- Performance on each GPU
- Memory bandwidth differences
- Compute capability impact

### 4.8 Experiment 7: Memory Usage Analysis

**Objective**: Measure memory consumption

```bash
# CPU memory
valgrind --tool=massif ./build/cnn_inference_cpu --model models/ --device cpu

# GPU memory
nvidia-smi --query-gpu=memory.used --format=csv --loop=1 &
./build/cnn_inference_gpu --model models/ --device gpu --batch 32
```

**Measure**:
- Peak host memory usage
- Peak device memory usage
- Memory vs. batch size relationship

---

## 5. Performance Analysis

### 5.1 Data Collection

Create benchmark script to automate data collection:

```bash
#!/bin/bash
# benchmark.sh

echo "batch,device,time_ms,memory_mb" > results.csv

for batch in 1 4 8 16 32 64; do
    # CPU sequential
    result=$(./build/cnn_inference_cpu --model models/ --device cpu --batch $batch)
    time=$(echo "$result" | grep "Inference completed" | awk '{print $4}')
    memory=$(echo "$result" | grep "Memory" | awk '{print $2}')
    echo "$batch,cpu_seq,$time,$memory" >> results.csv
    
    # Multicore CPU
    result=$(./build/cnn_inference_multicore --model models/ --device cpu --batch $batch)
    time=$(echo "$result" | grep "Inference completed" | awk '{print $4}')
    echo "$batch,cpu_multi,$time,$memory" >> results.csv
    
    # GPU
    result=$(./build/cnn_inference_gpu --model models/ --device gpu --batch $batch)
    time=$(echo "$result" | grep "Inference completed" | awk '{print $4}')
    memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
    echo "$batch,gpu,$time,$memory" >> results.csv
done
```

### 5.2 Visualization

Create plots for:

1. **Speedup Plot**
   - X-axis: Implementation (CPU seq, CPU multi, GPU)
   - Y-axis: Speedup (relative to CPU seq)
   - Bars for different batch sizes

2. **Throughput Plot**
   - X-axis: Batch size
   - Y-axis: Images/second
   - Lines for each implementation

3. **Memory Usage Plot**
   - X-axis: Batch size
   - Y-axis: Memory (MB/GB)
   - Lines for host and device memory

4. **Scaling Efficiency**
   - X-axis: Number of threads/cores
   - Y-axis: Speedup
   - Compare to linear speedup

### 5.3 Analysis Questions

Answer in your report:

1. **Speedup Analysis**
   - What is the GPU speedup? Is it expected?
   - Why does multicore CPU speedup saturate?
   - What limits GPU speedup?

2. **Batch Size Scaling**
   - How does throughput scale with batch size?
   - What is the optimal batch size?
   - Why does throughput saturate?

3. **Memory Analysis**
   - How much memory does each implementation use?
   - What is the memory bottleneck?
   - How does memory scale with batch size?

4. **Optimization Impact**
   - Which optimizations provided the most benefit?
   - What are the remaining bottlenecks?
   - What further optimizations are possible?

---

## 6. Results and Evaluation

### 6.1 Expected Results Structure

```
results/
├── performance.csv          # Raw timing data
├── memory_usage.csv         # Memory measurements
├── speedup_plot.png         # Speedup visualization
├── throughput_plot.png      # Throughput visualization
├── memory_plot.png          # Memory usage visualization
├── nvprof_report.txt        # GPU profiling report
└── analysis.md              # Written analysis
```

### 6.2 Results Template

#### 6.2.1 Performance Summary

| Implementation | Batch Size | Time (ms) | Throughput (img/s) | Speedup |
|---------------|------------|-----------|-------------------|---------|
| CPU Sequential | 1 | X | Y | 1.0× |
| CPU Sequential | 16 | X | Y | 1.0× |
| CPU Multicore | 1 | X | Y | Z× |
| CPU Multicore | 16 | X | Y | Z× |
| GPU | 1 | X | Y | Z× |
| GPU | 16 | X | Y | Z× |

#### 6.2.2 Memory Usage

| Implementation | Batch Size | Host Memory (MB) | Device Memory (MB) |
|---------------|------------|------------------|-------------------|
| CPU Sequential | 1 | X | N/A |
| CPU Sequential | 16 | X | N/A |
| GPU | 1 | X | Y |
| GPU | 16 | X | Y |

#### 6.2.3 Kernel Breakdown (GPU)

| Kernel | Time (ms) | Percentage | Memory Bandwidth (GB/s) |
|--------|-----------|------------|------------------------|
| conv2d | X | Y% | Z |
| maxpool | X | Y% | Z |
| gemm | X | Y% | Z |
| relu | X | Y% | Z |
| softmax | X | Y% | Z |

### 6.3 Performance Targets

Expected performance ranges (guidelines):

- **CPU Sequential**: 50-500 ms per image (batch=1)
- **CPU Multicore**: 10-100 ms per image (4-8 cores, batch=1)
  - Speedup: 4-8× vs. sequential
- **GPU**: 1-10 ms per image (batch=1)
  - Speedup: 10-100× vs. sequential
  - Speedup: 2-10× vs. multicore

**Note**: Actual performance depends on hardware, optimizations, and implementation quality.

---

## 7. Submission Requirements

### 7.1 Code Submission

**Repository Structure**:
```
GPU-CNN-Inference/
├── src/                    # Source code
│   ├── common/            # Utilities
│   ├── layers/            # Layer base classes
│   ├── cpu/               # CPU implementations
│   ├── gpu/               # GPU implementations
│   └── main.cpp           # Main entry point
├── Makefile               # Build configuration
├── README.md              # Project documentation
├── results/               # Experimental results
│   ├── performance.csv
│   ├── *.png              # Visualizations
│   └── analysis.md
└── report.pdf             # Final report
```

**Requirements**:
- ✅ Code must compile and run on CIMS machines
- ✅ No external dependencies beyond CUDA and standard libraries
- ✅ Clean, documented code
- ✅ Reproducible build process

### 7.2 Report Requirements

#### 7.2.1 Report Structure

1. **Introduction**
   - Project overview
   - Objectives
   - Model and dataset description

2. **Implementation Details**
   - Architecture overview
   - CPU sequential implementation
   - CPU multicore implementation (OpenMP)
   - GPU CUDA implementation
   - Key optimizations

3. **Experimental Setup**
   - Hardware specifications
   - Software environment
   - Benchmarking methodology

4. **Results**
   - Performance tables
   - Speedup analysis
   - Scalability analysis
   - Memory usage analysis

5. **Analysis and Discussion**
   - Performance bottlenecks
   - Optimization impact
   - Comparison with expectations
   - Lessons learned

6. **Conclusion**
   - Summary of findings
   - Future work

#### 7.2.2 Required Content

- ✅ Performance comparison tables
- ✅ Speedup plots (at least 3 visualizations)
- ✅ Analysis of bottlenecks
- ✅ Discussion of optimizations
- ✅ Comparison of CPU vs. GPU
- ✅ Scalability analysis (batch size, resolution)
- ✅ Memory analysis

#### 7.2.3 Code Quality

- ✅ Well-commented code
- ✅ Consistent coding style
- ✅ Modular design
- ✅ Error handling

### 7.3 Evaluation Criteria

**Performance (40%)**:
- Speedup achieved (GPU vs. CPU)
- Optimization quality
- Scalability demonstrated

**Correctness (20%)**:
- Implementation correctness
- Numerical accuracy
- Edge case handling

**Code Quality (20%)**:
- Code organization
- Documentation
- Reproducibility

**Report Quality (20%)**:
- Clarity of explanations
- Analysis depth
- Visualizations
- Discussion quality

### 7.4 Deadlines and Submission

- **Checkpoint 1**: CPU sequential implementation
- **Checkpoint 2**: CPU multicore implementation
- **Final Submission**: Complete implementation + report

**Submission Method**: GitHub repository + PDF report

---

## 8. Additional Resources

### 8.1 CUDA Documentation

- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- CUDA Toolkit Samples: `/usr/local/cuda/samples/` (on CIMS)

### 8.2 Profiling Tools

- **nvprof**: NVIDIA Profiler (legacy)
- **Nsight Compute**: Kernel profiling
- **Nsight Systems**: System-level profiling
- **nvidia-smi**: GPU monitoring

### 8.3 Optimization References

- NVIDIA Deep Learning Performance Guide
- CUDA Optimization Techniques
- Memory Access Pattern Optimization
- Kernel Fusion Strategies

### 8.4 Dataset

- Food-101: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- ImageNet: For model weights reference

---

## 9. Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Free unused device memory
   - Check for memory leaks

2. **Slow GPU Performance**
   - Check memory coalescing
   - Verify shared memory usage
   - Profile kernel execution

3. **Compilation Errors**
   - Verify CUDA version compatibility
   - Check include paths
   - Verify compiler flags

4. **Numerical Differences**
   - Check for floating-point precision issues
   - Verify reduction operations
   - Compare with reference implementation

---

## 10. Checklist

Before submission, verify:

- [ ] All three implementations complete (CPU seq, CPU multi, GPU)
- [ ] Code compiles on CIMS machines
- [ ] Benchmarking script works
- [ ] Results collected and validated
- [ ] Visualizations created
- [ ] Report written with all required sections
- [ ] Code documented
- [ ] Repository is clean and organized
- [ ] README.md is updated
- [ ] All files committed to GitHub

---

**Last Updated**: Based on project structure and common GPU course requirements

**Note**: Specific requirements may vary. Consult course materials and instructor guidelines for exact specifications.

