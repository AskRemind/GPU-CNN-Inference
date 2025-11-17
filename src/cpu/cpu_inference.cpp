#include "cpu_inference.h"
#include <iostream>
#include <algorithm>
#include <cstring>

CPUInference::CPUInference() : initialized_(false) {
}

CPUInference::~CPUInference() {
    layers_.clear();
}

bool CPUInference::initialize(const std::string& model_dir) {
    if (!model_loader_.loadWeights(model_dir)) {
        std::cerr << "Failed to load model weights" << std::endl;
        return false;
    }
    
    buildNetwork();
    initialized_ = true;
    
    std::cout << "CPU Inference Engine initialized" << std::endl;
    return true;
}

void CPUInference::buildNetwork() {
    layers_.clear();
    
    // VGG11 architecture
    // Conv layers
    layers_.push_back(std::make_unique<CPUConv2D>(
        model_loader_.getWeights("conv0.weight"),
        model_loader_.getWeights("conv0.bias"),
        3, 64, 3, 1, 1, "conv0"));
    layers_.push_back(std::make_unique<CPUReLU>("relu0"));
    layers_.push_back(std::make_unique<CPUMaxPool2D>(2, 2, "pool0"));
    
    layers_.push_back(std::make_unique<CPUConv2D>(
        model_loader_.getWeights("conv1.weight"),
        model_loader_.getWeights("conv1.bias"),
        64, 128, 3, 1, 1, "conv1"));
    layers_.push_back(std::make_unique<CPUReLU>("relu1"));
    layers_.push_back(std::make_unique<CPUMaxPool2D>(2, 2, "pool1"));
    
    layers_.push_back(std::make_unique<CPUConv2D>(
        model_loader_.getWeights("conv2.weight"),
        model_loader_.getWeights("conv2.bias"),
        128, 256, 3, 1, 1, "conv2"));
    layers_.push_back(std::make_unique<CPUReLU>("relu2"));
    
    layers_.push_back(std::make_unique<CPUConv2D>(
        model_loader_.getWeights("conv3.weight"),
        model_loader_.getWeights("conv3.bias"),
        256, 256, 3, 1, 1, "conv3"));
    layers_.push_back(std::make_unique<CPUReLU>("relu3"));
    layers_.push_back(std::make_unique<CPUMaxPool2D>(2, 2, "pool2"));
    
    layers_.push_back(std::make_unique<CPUConv2D>(
        model_loader_.getWeights("conv4.weight"),
        model_loader_.getWeights("conv4.bias"),
        256, 512, 3, 1, 1, "conv4"));
    layers_.push_back(std::make_unique<CPUReLU>("relu4"));
    
    layers_.push_back(std::make_unique<CPUConv2D>(
        model_loader_.getWeights("conv5.weight"),
        model_loader_.getWeights("conv5.bias"),
        512, 512, 3, 1, 1, "conv5"));
    layers_.push_back(std::make_unique<CPUReLU>("relu5"));
    layers_.push_back(std::make_unique<CPUMaxPool2D>(2, 2, "pool3"));
    
    layers_.push_back(std::make_unique<CPUConv2D>(
        model_loader_.getWeights("conv6.weight"),
        model_loader_.getWeights("conv6.bias"),
        512, 512, 3, 1, 1, "conv6"));
    layers_.push_back(std::make_unique<CPUReLU>("relu6"));
    
    layers_.push_back(std::make_unique<CPUConv2D>(
        model_loader_.getWeights("conv7.weight"),
        model_loader_.getWeights("conv7.bias"),
        512, 512, 3, 1, 1, "conv7"));
    layers_.push_back(std::make_unique<CPUReLU>("relu7"));
    layers_.push_back(std::make_unique<CPUMaxPool2D>(2, 2, "pool4"));
    
    // Fully connected layers
    layers_.push_back(std::make_unique<CPULinear>(
        model_loader_.getWeights("fc0.weight"),
        model_loader_.getWeights("fc0.bias"),
        25088, 4096, "fc0"));
    layers_.push_back(std::make_unique<CPUReLU>("relu8"));
    
    layers_.push_back(std::make_unique<CPULinear>(
        model_loader_.getWeights("fc1.weight"),
        model_loader_.getWeights("fc1.bias"),
        4096, 4096, "fc1"));
    layers_.push_back(std::make_unique<CPUReLU>("relu9"));
    
    layers_.push_back(std::make_unique<CPULinear>(
        model_loader_.getWeights("fc2.weight"),
        model_loader_.getWeights("fc2.bias"),
        4096, 1000, "fc2"));
    
    layers_.push_back(std::make_unique<CPUSoftmax>("softmax"));
    
    std::cout << "Network built with " << layers_.size() << " layers" << std::endl;
}

bool CPUInference::infer(const float* image_data, float* output, int batch_size) {
    if (!initialized_) {
        std::cerr << "Error: Inference engine not initialized" << std::endl;
        return false;
    }
    
    if (batch_size <= 0) {
        std::cerr << "Error: Invalid batch size: " << batch_size << std::endl;
        return false;
    }
    
    // Use ping-pong buffers to avoid unnecessary memory copies
    // Maximum intermediate buffer size: [batch_size, 64, 224, 224]
    // For batch_size up to 32, max is: 32 * 64 * 224 * 224 = 102,760,448
    // We'll allocate buffer dynamically if needed, or use static for common sizes
    static std::vector<float> bufferA(3211264);  // For batch=1: [1, 64, 224, 224]
    static std::vector<float> bufferB(3211264);
    static int current_buffer_size = 3211264;  // Track current buffer size
    
    // Calculate required buffer size
    int max_buffer_size = batch_size * 64 * 224 * 224;  // After conv0
    if (max_buffer_size > current_buffer_size) {
        // Need larger buffers for larger batches - resize directly
        bufferA.resize(max_buffer_size);
        bufferB.resize(max_buffer_size);
        current_buffer_size = max_buffer_size;
    }
    
    // Input shape: [batch_size, 3, 224, 224]
    std::vector<int> current_shape = {batch_size, 3, 224, 224};
    
    // Debug: Verify batch_size
    std::cout << "[CPU Debug] batch_size=" << batch_size 
              << ", input_size=" << (batch_size * 3 * 224 * 224) 
              << ", max_buffer_size=" << max_buffer_size << std::endl;
    
    // Copy input images to first buffer
    std::memcpy(bufferA.data(), image_data, batch_size * 3 * 224 * 224 * sizeof(float));
    
    // Use pointers to swap buffers without copying data
    float* input_ptr = bufferA.data();
    float* output_ptr = bufferB.data();
    
    // Forward pass through all layers
    for (size_t i = 0; i < layers_.size(); i++) {
        std::vector<int> next_shape;
        next_shape = layers_[i]->getOutputShape(current_shape);
        
        if (i == layers_.size() - 1) {
            // Last layer: output directly to provided buffer
            // Output shape: [batch_size, 1000]
            if (!layers_[i]->forward(input_ptr, current_shape,
                                    output, next_shape)) {
                return false;
            }
        } else {
            // Intermediate layers: write to output buffer
            if (!layers_[i]->forward(input_ptr, current_shape,
                                    output_ptr, next_shape)) {
                return false;
            }
            // Swap pointers instead of copying data (ping-pong buffers)
            std::swap(input_ptr, output_ptr);
            current_shape = next_shape;
        }
    }
    
    return true;
}

