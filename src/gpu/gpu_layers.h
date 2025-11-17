#ifndef GPU_LAYERS_H
#define GPU_LAYERS_H

#include "gpu_conv2d.h"
#include "../layers/layer_base.h"
#include <string>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

/**
 * GPU implementation of ReLU activation using CUDA
 */
class GPURELU : public Layer {
public:
    GPURELU(const std::string& name) : name_(name) {}
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override;
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override {
        return input_shape;
    }
    
    std::string getName() const override { return name_; }
    
private:
    std::string name_;
};

/**
 * GPU implementation of MaxPool2D using CUDA
 */
class GPUMaxPool2D : public Layer {
public:
    GPUMaxPool2D(int kernel_size, int stride, const std::string& name)
        : kernel_size_(kernel_size), stride_(stride), name_(name) {}
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override;
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override {
        int batch = input_shape[0];
        int channels = input_shape[1];
        int in_h = input_shape[2];
        int in_w = input_shape[3];
        int out_h = in_h / stride_;
        int out_w = in_w / stride_;
        return {batch, channels, out_h, out_w};
    }
    
    std::string getName() const override { return name_; }
    
private:
    int kernel_size_;
    int stride_;
    std::string name_;
};

/**
 * GPU implementation of Fully Connected (Linear) layer using CUDA
 */
class GPULinear : public Layer {
public:
    GPULinear(const float* weights, const float* bias,
             int in_features, int out_features,
             const std::string& name);
    
    ~GPULinear();
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override;
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override {
        int batch = input_shape[0];
        return {batch, out_features_};
    }
    
    std::string getName() const override { return name_; }
    
private:
    const float* host_weights_;  // [out_features, in_features]
    const float* host_bias_;     // [out_features]
    float* device_weights_;
    float* device_bias_;
    int in_features_;
    int out_features_;
    std::string name_;
    
    void uploadWeights();
    void freeDeviceMemory();
};

/**
 * GPU implementation of Softmax using CUDA
 */
class GPUSoftmax : public Layer {
public:
    GPUSoftmax(const std::string& name) : name_(name) {}
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override;
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override {
        return input_shape;
    }
    
    std::string getName() const override { return name_; }
    
private:
    std::string name_;
};

// CUDA kernel declarations
extern "C" void launchRELUKernel(const float* input, float* output, int size);
extern "C" void launchMaxPool2DKernel(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w, int out_h, int out_w,
    int kernel_size, int stride);
extern "C" void launchLinearKernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch, int in_features, int out_features);
extern "C" void launchSoftmaxKernel(const float* input, float* output,
                                    int batch, int num_classes);

#endif // GPU_LAYERS_H

