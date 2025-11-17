#ifndef CPU_MULTICORE_LAYERS_H
#define CPU_MULTICORE_LAYERS_H

#include "cpu_multicore_conv2d.h"
#include "../layers/layer_base.h"
#include <string>
#include <cmath>
#include <limits>
#include <omp.h>

/**
 * CPU Multicore implementation of ReLU activation using OpenMP
 */
class CPUMulticoreReLU : public Layer {
public:
    CPUMulticoreReLU(const std::string& name) : name_(name) {}
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override {
        output_shape = input_shape;
        int size = 1;
        for (int dim : input_shape) size *= dim;
        
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            output[i] = std::max(0.0f, input[i]);
        }
        return true;
    }
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override {
        return input_shape;
    }
    
    std::string getName() const override { return name_; }
    
private:
    std::string name_;
};

/**
 * CPU Multicore implementation of MaxPool2D using OpenMP
 */
class CPUMulticoreMaxPool2D : public Layer {
public:
    CPUMulticoreMaxPool2D(int kernel_size, int stride, const std::string& name)
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
 * CPU Multicore implementation of Fully Connected (Linear) layer using OpenMP
 */
class CPUMulticoreLinear : public Layer {
public:
    CPUMulticoreLinear(const float* weights, const float* bias,
                      int in_features, int out_features,
                      const std::string& name);
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override;
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override {
        int batch = input_shape[0];
        return {batch, out_features_};
    }
    
    std::string getName() const override { return name_; }
    
private:
    const float* weights_;  // [out_features, in_features]
    const float* bias_;     // [out_features]
    int in_features_;
    int out_features_;
    std::string name_;
};

/**
 * CPU Multicore implementation of Softmax using OpenMP
 */
class CPUMulticoreSoftmax : public Layer {
public:
    CPUMulticoreSoftmax(const std::string& name) : name_(name) {}
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override;
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override {
        return input_shape;
    }
    
    std::string getName() const override { return name_; }
    
private:
    std::string name_;
};

#endif // CPU_MULTICORE_LAYERS_H

