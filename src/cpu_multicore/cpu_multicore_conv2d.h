#ifndef CPU_MULTICORE_CONV2D_H
#define CPU_MULTICORE_CONV2D_H

#include "../layers/layer_base.h"
#include <string>

/**
 * CPU Multicore implementation of 2D Convolution layer using OpenMP
 * Supports padding and stride with parallel processing
 */
class CPUMulticoreConv2D : public Layer {
public:
    CPUMulticoreConv2D(const float* weights, const float* bias,
                       int in_channels, int out_channels,
                       int kernel_size, int padding, int stride,
                       const std::string& name);
    
    bool forward(const float* input, const std::vector<int>& input_shape,
                float* output, std::vector<int>& output_shape) override;
    
    std::vector<int> getOutputShape(const std::vector<int>& input_shape) const override;
    
    std::string getName() const override { return name_; }
    
private:
    const float* weights_;      // [out_channels, in_channels, kernel_h, kernel_w]
    const float* bias_;         // [out_channels]
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int padding_;
    int stride_;
    std::string name_;
    
    // Helper function for convolution operation with OpenMP parallelization
    void convolve(const float* input, float* output,
                 int batch, int in_h, int in_w, int out_h, int out_w);
};

#endif // CPU_MULTICORE_CONV2D_H

