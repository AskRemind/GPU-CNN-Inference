#ifndef GPU_CONV2D_H
#define GPU_CONV2D_H

#include "../layers/layer_base.h"
#include <string>
#include <cuda_runtime.h>

/**
 * GPU implementation of 2D Convolution layer using CUDA
 * Supports padding and stride with GPU parallelization
 */
class GPUConv2D : public Layer {
public:
    GPUConv2D(const float* weights, const float* bias,
              int in_channels, int out_channels,
              int kernel_size, int padding, int stride,
              const std::string& name);
    
    ~GPUConv2D();
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override;
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override;
    
    std::string getName() const override { return name_; }
    
private:
    const float* host_weights_;      // [out_channels, in_channels, kernel_h, kernel_w]
    const float* host_bias_;         // [out_channels]
    float* device_weights_;
    float* device_bias_;
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int padding_;
    int stride_;
    std::string name_;
    
    void uploadWeights();
    void freeDeviceMemory();
};

// CUDA kernel declaration
extern "C" void launchConv2DKernel(
    const float* input, const float* weights, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_size, int padding, int stride);

#endif // GPU_CONV2D_H

