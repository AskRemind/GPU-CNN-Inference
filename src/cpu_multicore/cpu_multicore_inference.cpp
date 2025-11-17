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
    
    // Input shape: [1, 3, 224, 224] (batch=1)
    std::vector<int> current_shape = {1, 3, 224, 224};
    std::vector<float> current_data(1 * 3 * 224 * 224);
    std::memcpy(current_data.data(), image_data, 1 * 3 * 224 * 224 * sizeof(float));
    
    std::vector<float> next_data;
    
    // Forward pass through all layers
    for (size_t i = 0; i < layers_.size(); i++) {
        std::vector<int> next_shape;
        next_shape = layers_[i]->getOutputShape(current_shape);
        
        int next_size = 1;
        for (int dim : next_shape) next_size *= dim;
        
        if (i == layers_.size() - 1) {
            // Last layer: output directly to provided buffer
            if (!layers_[i]->forward(current_data.data(), current_shape,
                                    output, next_shape)) {
                return false;
            }
        } else {
            // Intermediate layers: use temporary buffer
            next_data.resize(next_size);
            if (!layers_[i]->forward(current_data.data(), current_shape,
                                    next_data.data(), next_shape)) {
                return false;
            }
            current_data = next_data;
            current_shape = next_shape;
        }
    }
    
    return true;
}

