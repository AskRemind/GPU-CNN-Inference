#include "gpu_inference.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

GPUInference::GPUInference() : initialized_(false) {
    // Initialize CUDA device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
}

GPUInference::~GPUInference() {
    layers_.clear();
}

bool GPUInference::initialize(const std::string& model_dir) {
    if (!model_loader_.loadWeights(model_dir)) {
        std::cerr << "Failed to load model weights" << std::endl;
        return false;
    }
    
    buildNetwork();
    initialized_ = true;
    
    std::cout << "GPU Inference Engine initialized" << std::endl;
    return true;
}

void GPUInference::buildNetwork() {
    layers_.clear();
    
    // VGG11 architecture (same as CPU versions)
    // Conv layers
    layers_.push_back(std::make_unique<GPUConv2D>(
        model_loader_.getWeights("conv0.weight"),
        model_loader_.getWeights("conv0.bias"),
        3, 64, 3, 1, 1, "conv0"));
    layers_.push_back(std::make_unique<GPURELU>("relu0"));
    layers_.push_back(std::make_unique<GPUMaxPool2D>(2, 2, "pool0"));
    
    layers_.push_back(std::make_unique<GPUConv2D>(
        model_loader_.getWeights("conv1.weight"),
        model_loader_.getWeights("conv1.bias"),
        64, 128, 3, 1, 1, "conv1"));
    layers_.push_back(std::make_unique<GPURELU>("relu1"));
    layers_.push_back(std::make_unique<GPUMaxPool2D>(2, 2, "pool1"));
    
    layers_.push_back(std::make_unique<GPUConv2D>(
        model_loader_.getWeights("conv2.weight"),
        model_loader_.getWeights("conv2.bias"),
        128, 256, 3, 1, 1, "conv2"));
    layers_.push_back(std::make_unique<GPURELU>("relu2"));
    
    layers_.push_back(std::make_unique<GPUConv2D>(
        model_loader_.getWeights("conv3.weight"),
        model_loader_.getWeights("conv3.bias"),
        256, 256, 3, 1, 1, "conv3"));
    layers_.push_back(std::make_unique<GPURELU>("relu3"));
    layers_.push_back(std::make_unique<GPUMaxPool2D>(2, 2, "pool2"));
    
    layers_.push_back(std::make_unique<GPUConv2D>(
        model_loader_.getWeights("conv4.weight"),
        model_loader_.getWeights("conv4.bias"),
        256, 512, 3, 1, 1, "conv4"));
    layers_.push_back(std::make_unique<GPURELU>("relu4"));
    
    layers_.push_back(std::make_unique<GPUConv2D>(
        model_loader_.getWeights("conv5.weight"),
        model_loader_.getWeights("conv5.bias"),
        512, 512, 3, 1, 1, "conv5"));
    layers_.push_back(std::make_unique<GPURELU>("relu5"));
    layers_.push_back(std::make_unique<GPUMaxPool2D>(2, 2, "pool3"));
    
    layers_.push_back(std::make_unique<GPUConv2D>(
        model_loader_.getWeights("conv6.weight"),
        model_loader_.getWeights("conv6.bias"),
        512, 512, 3, 1, 1, "conv6"));
    layers_.push_back(std::make_unique<GPURELU>("relu6"));
    
    layers_.push_back(std::make_unique<GPUConv2D>(
        model_loader_.getWeights("conv7.weight"),
        model_loader_.getWeights("conv7.bias"),
        512, 512, 3, 1, 1, "conv7"));
    layers_.push_back(std::make_unique<GPURELU>("relu7"));
    layers_.push_back(std::make_unique<GPUMaxPool2D>(2, 2, "pool4"));
    
    // Fully connected layers
    layers_.push_back(std::make_unique<GPULinear>(
        model_loader_.getWeights("fc0.weight"),
        model_loader_.getWeights("fc0.bias"),
        25088, 4096, "fc0"));
    layers_.push_back(std::make_unique<GPURELU>("relu8"));
    
    layers_.push_back(std::make_unique<GPULinear>(
        model_loader_.getWeights("fc1.weight"),
        model_loader_.getWeights("fc1.bias"),
        4096, 4096, "fc1"));
    layers_.push_back(std::make_unique<GPURELU>("relu9"));
    
    layers_.push_back(std::make_unique<GPULinear>(
        model_loader_.getWeights("fc2.weight"),
        model_loader_.getWeights("fc2.bias"),
        4096, 1000, "fc2"));
    
    layers_.push_back(std::make_unique<GPUSoftmax>("softmax"));
    
    std::cout << "Network built with " << layers_.size() << " layers" << std::endl;
}

bool GPUInference::infer(const float* image_data, float* output) {
    if (!initialized_) {
        std::cerr << "Error: Inference engine not initialized" << std::endl;
        return false;
    }
    
    // Use ping-pong buffers on GPU to avoid Host-GPU-Host round trips
    // Maximum intermediate buffer size: [1, 64, 224, 224] = 3,211,264 (after conv0)
    const int max_buffer_size = 3211264;
    
    // Allocate GPU buffers (static for reuse across inference calls)
    static float* device_buffer_A = nullptr;
    static float* device_buffer_B = nullptr;
    static bool buffers_allocated = false;
    
    #define CHECK_CUDA_RET(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                          << " - " << cudaGetErrorString(err) << std::endl; \
                return false; \
            } \
        } while(0)
    
    if (!buffers_allocated) {
        CHECK_CUDA_RET(cudaMalloc(&device_buffer_A, max_buffer_size * sizeof(float)));
        CHECK_CUDA_RET(cudaMalloc(&device_buffer_B, max_buffer_size * sizeof(float)));
        buffers_allocated = true;
    }
    
    // Input shape: [1, 3, 224, 224] (batch=1)
    std::vector<int> current_shape = {1, 3, 224, 224};
    
    // Copy input image to GPU (ONCE at the beginning)
    CHECK_CUDA_RET(cudaMemcpy(device_buffer_A, image_data,
                             (1 * 3 * 224 * 224) * sizeof(float),
                             cudaMemcpyHostToDevice));
    
    // Use pointers to swap buffers without copying data
    float* device_input_ptr = device_buffer_A;
    float* device_output_ptr = device_buffer_B;
    
    // Forward pass through all layers (all data stays on GPU)
    for (size_t i = 0; i < layers_.size(); i++) {
        std::vector<int> next_shape;
        next_shape = layers_[i]->getOutputShape(current_shape);
        
        if (i == layers_.size() - 1) {
            // Last layer: allocate temporary GPU buffer for final output
            int final_size = 1000; // [1, 1000]
            float* final_device_output;
            CHECK_CUDA_RET(cudaMalloc(&final_device_output, final_size * sizeof(float)));
            
            // Last layer forward pass (all on GPU)
            if (!layers_[i]->forward(device_input_ptr, current_shape,
                                    final_device_output, next_shape)) {
                cudaFree(final_device_output);
                return false;
            }
            
            // Copy final result from GPU to CPU (ONCE at the end)
            CHECK_CUDA_RET(cudaMemcpy(output, final_device_output,
                                     final_size * sizeof(float),
                                     cudaMemcpyDeviceToHost));
            
            cudaFree(final_device_output);
        } else {
            // Intermediate layers: ping-pong between GPU buffers
            // Note: GPU layers expect GPU pointers, even though Layer interface uses const float*
            if (!layers_[i]->forward(device_input_ptr, current_shape,
                                    device_output_ptr, next_shape)) {
                return false;
            }
            
            // Swap pointers instead of copying data (no Host-GPU-Host round trip)
            std::swap(device_input_ptr, device_output_ptr);
            current_shape = next_shape;
        }
    }
    
    return true;
}

