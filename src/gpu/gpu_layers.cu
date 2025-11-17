#include "gpu_layers.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <cfloat>

#define CHECK_CUDA_RET(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

// CUDA kernel for ReLU
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" void launchRELUKernel(const float* input, float* output, int size) {
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<num_blocks, threads_per_block>>>(input, output, size);
    cudaDeviceSynchronize();
}

bool GPURELU::forward(const float* input, const std::vector<int>& input_shape,
                     float* output, std::vector<int>& output_shape) {
    output_shape = input_shape;
    int size = 1;
    for (int dim : input_shape) size *= dim;
    
    float* device_input;
    float* device_output;
    
    CHECK_CUDA_RET(cudaMalloc(&device_input, size * sizeof(float)));
    CHECK_CUDA_RET(cudaMalloc(&device_output, size * sizeof(float)));
    
    CHECK_CUDA_RET(cudaMemcpy(device_input, input,
                             size * sizeof(float), cudaMemcpyHostToDevice));
    
    launchRELUKernel(device_input, device_output, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }
    
    CHECK_CUDA_RET(cudaMemcpy(output, device_output,
                             size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(device_input);
    cudaFree(device_output);
    
    return true;
}

// CUDA kernel for MaxPool2D
__global__ void maxpool2d_kernel(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w, int out_h, int out_w,
    int kernel_size, int stride) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch * channels * out_h * out_w;
    
    if (idx >= total_output) return;
    
    int b = idx / (channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % channels;
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;
    
    float max_val = -FLT_MAX;
    
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;
            
            if (ih < in_h && iw < in_w) {
                int input_idx = b * channels * in_h * in_w +
                                c * in_h * in_w + ih * in_w + iw;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }
    
    output[idx] = max_val;
}

extern "C" void launchMaxPool2DKernel(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w, int out_h, int out_w,
    int kernel_size, int stride) {
    
    int total_output = batch * channels * out_h * out_w;
    int threads_per_block = 256;
    int num_blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    maxpool2d_kernel<<<num_blocks, threads_per_block>>>(
        input, output, batch, channels, in_h, in_w, out_h, out_w,
        kernel_size, stride);
    
    cudaDeviceSynchronize();
}

bool GPUMaxPool2D::forward(const float* input, const std::vector<int>& input_shape,
                          float* output, std::vector<int>& output_shape) {
    int batch = input_shape[0];
    int channels = input_shape[1];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    int out_h = in_h / stride_;
    int out_w = in_w / stride_;
    
    output_shape = {batch, channels, out_h, out_w};
    
    int input_size = batch * channels * in_h * in_w;
    int output_size = batch * channels * out_h * out_w;
    
    float* device_input;
    float* device_output;
    
    CHECK_CUDA_RET(cudaMalloc(&device_input, input_size * sizeof(float)));
    CHECK_CUDA_RET(cudaMalloc(&device_output, output_size * sizeof(float)));
    
    CHECK_CUDA_RET(cudaMemcpy(device_input, input,
                             input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    launchMaxPool2DKernel(device_input, device_output,
                         batch, channels, in_h, in_w, out_h, out_w,
                         kernel_size_, stride_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }
    
    CHECK_CUDA_RET(cudaMemcpy(output, device_output,
                             output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(device_input);
    cudaFree(device_output);
    
    return true;
}

// CUDA kernel for Linear (Matrix Multiplication)
__global__ void linear_kernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch, int in_features, int out_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch * out_features;
    
    if (idx >= total_output) return;
    
    int b = idx / out_features;
    int of = idx % out_features;
    
    float sum = bias[of];
    
    for (int inf = 0; inf < in_features; inf++) {
        int input_idx = b * in_features + inf;
        int weight_idx = of * in_features + inf;
        sum += input[input_idx] * weights[weight_idx];
    }
    
    output[idx] = sum;
}

extern "C" void launchLinearKernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch, int in_features, int out_features) {
    
    int total_output = batch * out_features;
    int threads_per_block = 256;
    int num_blocks = (total_output + threads_per_block - 1) / threads_per_block;
    
    linear_kernel<<<num_blocks, threads_per_block>>>(
        input, weights, bias, output,
        batch, in_features, out_features);
    
    cudaDeviceSynchronize();
}

GPULinear::GPULinear(const float* weights, const float* bias,
                     int in_features, int out_features,
                     const std::string& name)
    : host_weights_(weights), host_bias_(bias),
      device_weights_(nullptr), device_bias_(nullptr),
      in_features_(in_features), out_features_(out_features),
      name_(name) {
    uploadWeights();
}

GPULinear::~GPULinear() {
    freeDeviceMemory();
}

void GPULinear::uploadWeights() {
    int weights_size = out_features_ * in_features_;
    int bias_size = out_features_;
    
    cudaMalloc(&device_weights_, weights_size * sizeof(float));
    cudaMalloc(&device_bias_, bias_size * sizeof(float));
    
    cudaMemcpy(device_weights_, host_weights_,
              weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias_, host_bias_,
              bias_size * sizeof(float), cudaMemcpyHostToDevice);
}

void GPULinear::freeDeviceMemory() {
    if (device_weights_) {
        cudaFree(device_weights_);
        device_weights_ = nullptr;
    }
    if (device_bias_) {
        cudaFree(device_bias_);
        device_bias_ = nullptr;
    }
}

bool GPULinear::forward(const float* input, const std::vector<int>& input_shape,
                       float* output, std::vector<int>& output_shape) {
    int batch = input_shape[0];
    int features = 1;
    for (size_t i = 1; i < input_shape.size(); i++) {
        features *= input_shape[i];
    }
    
    if (features != in_features_) {
        std::cerr << "Error: Input features mismatch in " << name_ << std::endl;
        return false;
    }
    
    output_shape = {batch, out_features_};
    
    int input_size = batch * in_features_;
    int output_size = batch * out_features_;
    
    float* device_input;
    float* device_output;
    
    CHECK_CUDA_RET(cudaMalloc(&device_input, input_size * sizeof(float)));
    CHECK_CUDA_RET(cudaMalloc(&device_output, output_size * sizeof(float)));
    
    CHECK_CUDA_RET(cudaMemcpy(device_input, input,
                             input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    launchLinearKernel(device_input, device_weights_, device_bias_, device_output,
                      batch, in_features_, out_features_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }
    
    CHECK_CUDA_RET(cudaMemcpy(output, device_output,
                             output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(device_input);
    cudaFree(device_output);
    
    return true;
}

// CUDA kernel for Softmax
__global__ void softmax_kernel(const float* input, float* output,
                               int batch, int num_classes) {
    int b = blockIdx.x;
    if (b >= batch) return;
    
    // Find max for numerical stability
    float max_val = input[b * num_classes];
    for (int i = 1; i < num_classes; i++) {
        max_val = fmaxf(max_val, input[b * num_classes + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        float exp_val = expf(input[b * num_classes + i] - max_val);
        output[b * num_classes + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < num_classes; i++) {
        output[b * num_classes + i] /= sum;
    }
}

extern "C" void launchSoftmaxKernel(const float* input, float* output,
                                    int batch, int num_classes) {
    softmax_kernel<<<batch, 1>>>(input, output, batch, num_classes);
    cudaDeviceSynchronize();
}

bool GPUSoftmax::forward(const float* input, const std::vector<int>& input_shape,
                        float* output, std::vector<int>& output_shape) {
    output_shape = input_shape;
    
    int batch = input_shape[0];
    int num_classes = 1;
    for (size_t i = 1; i < input_shape.size(); i++) {
        num_classes *= input_shape[i];
    }
    
    int size = batch * num_classes;
    
    float* device_input;
    float* device_output;
    
    CHECK_CUDA_RET(cudaMalloc(&device_input, size * sizeof(float)));
    CHECK_CUDA_RET(cudaMalloc(&device_output, size * sizeof(float)));
    
    CHECK_CUDA_RET(cudaMemcpy(device_input, input,
                             size * sizeof(float), cudaMemcpyHostToDevice));
    
    launchSoftmaxKernel(device_input, device_output, batch, num_classes);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }
    
    CHECK_CUDA_RET(cudaMemcpy(output, device_output,
                             size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(device_input);
    cudaFree(device_output);
    
    return true;
}

