#include "cpu_multicore_inference.h"
#include <iostream>
#include <algorithm>
#include <cstring>

CPUMulticoreInference::CPUMulticoreInference() : initialized_(false) {
}

CPUMulticoreInference::~CPUMulticoreInference() {
    layers_.clear();
}

bool CPUMulticoreInference::initialize(const std::string& model_dir) {
    if (!model_loader_.loadWeights(model_dir)) {
        std::cerr << "Failed to load model weights" << std::endl;
        return false;
    }
    
    buildNetwork();
    initialized_ = true;
    
    std::cout << "CPU Multicore Inference Engine initialized" << std::endl;
    return true;
}

void CPUMulticoreInference::buildNetwork() {
    layers_.clear();
    
    // VGG11 architecture (same as CPU sequential)
    // Conv layers
    layers_.push_back(std::make_unique<CPUMulticoreConv2D>(
        model_loader_.getWeights("conv0.weight"),
        model_loader_.getWeights("conv0.bias"),
        3, 64, 3, 1, 1, "conv0"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu0"));
    layers_.push_back(std::make_unique<CPUMulticoreMaxPool2D>(2, 2, "pool0"));
    
    layers_.push_back(std::make_unique<CPUMulticoreConv2D>(
        model_loader_.getWeights("conv1.weight"),
        model_loader_.getWeights("conv1.bias"),
        64, 128, 3, 1, 1, "conv1"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu1"));
    layers_.push_back(std::make_unique<CPUMulticoreMaxPool2D>(2, 2, "pool1"));
    
    layers_.push_back(std::make_unique<CPUMulticoreConv2D>(
        model_loader_.getWeights("conv2.weight"),
        model_loader_.getWeights("conv2.bias"),
        128, 256, 3, 1, 1, "conv2"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu2"));
    
    layers_.push_back(std::make_unique<CPUMulticoreConv2D>(
        model_loader_.getWeights("conv3.weight"),
        model_loader_.getWeights("conv3.bias"),
        256, 256, 3, 1, 1, "conv3"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu3"));
    layers_.push_back(std::make_unique<CPUMulticoreMaxPool2D>(2, 2, "pool2"));
    
    layers_.push_back(std::make_unique<CPUMulticoreConv2D>(
        model_loader_.getWeights("conv4.weight"),
        model_loader_.getWeights("conv4.bias"),
        256, 512, 3, 1, 1, "conv4"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu4"));
    
    layers_.push_back(std::make_unique<CPUMulticoreConv2D>(
        model_loader_.getWeights("conv5.weight"),
        model_loader_.getWeights("conv5.bias"),
        512, 512, 3, 1, 1, "conv5"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu5"));
    layers_.push_back(std::make_unique<CPUMulticoreMaxPool2D>(2, 2, "pool3"));
    
    layers_.push_back(std::make_unique<CPUMulticoreConv2D>(
        model_loader_.getWeights("conv6.weight"),
        model_loader_.getWeights("conv6.bias"),
        512, 512, 3, 1, 1, "conv6"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu6"));
    
    layers_.push_back(std::make_unique<CPUMulticoreConv2D>(
        model_loader_.getWeights("conv7.weight"),
        model_loader_.getWeights("conv7.bias"),
        512, 512, 3, 1, 1, "conv7"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu7"));
    layers_.push_back(std::make_unique<CPUMulticoreMaxPool2D>(2, 2, "pool4"));
    
    // Fully connected layers
    layers_.push_back(std::make_unique<CPUMulticoreLinear>(
        model_loader_.getWeights("fc0.weight"),
        model_loader_.getWeights("fc0.bias"),
        25088, 4096, "fc0"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu8"));
    
    layers_.push_back(std::make_unique<CPUMulticoreLinear>(
        model_loader_.getWeights("fc1.weight"),
        model_loader_.getWeights("fc1.bias"),
        4096, 4096, "fc1"));
    layers_.push_back(std::make_unique<CPUMulticoreReLU>("relu9"));
    
    layers_.push_back(std::make_unique<CPUMulticoreLinear>(
        model_loader_.getWeights("fc2.weight"),
        model_loader_.getWeights("fc2.bias"),
        4096, 1000, "fc2"));
    
    layers_.push_back(std::make_unique<CPUMulticoreSoftmax>("softmax"));
    
    std::cout << "Network built with " << layers_.size() << " layers" << std::endl;
}

bool CPUMulticoreInference::infer(const float* image_data, float* output) {
    if (!initialized_) {
        std::cerr << "Error: Inference engine not initialized" << std::endl;
        return false;
    }
    
    // Use ping-pong buffers to avoid unnecessary memory copies
    // Maximum intermediate buffer size: [1, 64, 224, 224] = 3,211,264 (after conv0)
    // Using static buffers allocated once, reused for all inference calls
    static std::vector<float> bufferA(3211264);
    static std::vector<float> bufferB(3211264);
    
    // Input shape: [1, 3, 224, 224] (batch=1)
    std::vector<int> current_shape = {1, 3, 224, 224};
    
    // Copy input image to first buffer
    std::memcpy(bufferA.data(), image_data, 1 * 3 * 224 * 224 * sizeof(float));
    
    // Use pointers to swap buffers without copying data
    float* input_ptr = bufferA.data();
    float* output_ptr = bufferB.data();
    
    // Forward pass through all layers
    for (size_t i = 0; i < layers_.size(); i++) {
        std::vector<int> next_shape;
        next_shape = layers_[i]->getOutputShape(current_shape);
        
        if (i == layers_.size() - 1) {
            // Last layer: output directly to provided buffer
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

