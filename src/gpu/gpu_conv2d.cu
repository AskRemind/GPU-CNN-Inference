#include "gpu_conv2d.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
        } \
    } while(0)
    
#define CHECK_CUDA_RET(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

// CUDA kernel for 2D convolution
__global__ void conv2d_kernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_size, int padding, int stride) {
    
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch * out_channels * out_h * out_w;
    
    if (idx >= total_output) return;
    
    // Calculate output position
    int b = idx / (out_channels * out_h * out_w);
    int oc = (idx / (out_h * out_w)) % out_channels;
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;
    
    // Accumulate convolution result
    float sum = bias[oc];
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = b * in_channels * in_h * in_w +
                                    ic * in_h * in_w + ih * in_w + iw;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size +
                                     kh * kernel_size + kw;
                    
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}

// Kernel launcher
extern "C" void launchConv2DKernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_size, int padding, int stride) {
    
    int total_output = batch * out_channels * out_h * out_w;
    int threads_per_block = 256;
    int num_blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    conv2d_kernel<<<num_blocks, threads_per_block>>>(
        input, weights, bias, output,
        batch, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_size, padding, stride);
    
    cudaDeviceSynchronize();
}

GPUConv2D::GPUConv2D(const float* weights, const float* bias,
                     int in_channels, int out_channels,
                     int kernel_size, int padding, int stride,
                     const std::string& name)
    : host_weights_(weights), host_bias_(bias),
      device_weights_(nullptr), device_bias_(nullptr),
      in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), padding_(padding), stride_(stride),
      name_(name) {
    uploadWeights();
}

GPUConv2D::~GPUConv2D() {
    freeDeviceMemory();
}

void GPUConv2D::uploadWeights() {
    int weights_size = out_channels_ * in_channels_ * kernel_size_ * kernel_size_;
    int bias_size = out_channels_;
    
    cudaMalloc(&device_weights_, weights_size * sizeof(float));
    cudaMalloc(&device_bias_, bias_size * sizeof(float));
    
    cudaMemcpy(device_weights_, host_weights_,
              weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias_, host_bias_,
              bias_size * sizeof(float), cudaMemcpyHostToDevice);
}

void GPUConv2D::freeDeviceMemory() {
    if (device_weights_) {
        cudaFree(device_weights_);
        device_weights_ = nullptr;
    }
    if (device_bias_) {
        cudaFree(device_bias_);
        device_bias_ = nullptr;
    }
}

std::vector<int> GPUConv2D::getOutputShape(const std::vector<int>& input_shape) const {
    int batch = input_shape[0];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    
    int out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    return {batch, out_channels_, out_h, out_w};
}

bool GPUConv2D::forward(const float* input, const std::vector<int>& input_shape,
                        float* output, std::vector<int>& output_shape) {
    int batch = input_shape[0];
    int in_channels = input_shape[1];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    
    if (in_channels != in_channels_) {
        std::cerr << "Error: Input channels mismatch in " << name_ << std::endl;
        return false;
    }
    
    int out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;
    output_shape = {batch, out_channels_, out_h, out_w};
    
    // Input and output pointers are already on GPU (managed by GPUInference)
    // No cudaMalloc, cudaMemcpy, or cudaFree needed here
    // Cast to non-const for kernel launch (kernel doesn't modify input)
    float* device_input = const_cast<float*>(input);
    float* device_output = output;
    
    // Launch kernel directly (all data already on GPU)
    launchConv2DKernel(device_input, device_weights_, device_bias_, device_output,
                      batch, in_channels_, out_channels_,
                      in_h, in_w, out_h, out_w,
                      kernel_size_, padding_, stride_);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

