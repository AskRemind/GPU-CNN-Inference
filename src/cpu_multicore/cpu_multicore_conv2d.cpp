#include "cpu_multicore_conv2d.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <omp.h>

CPUMulticoreConv2D::CPUMulticoreConv2D(const float* weights, const float* bias,
                                       int in_channels, int out_channels,
                                       int kernel_size, int padding, int stride,
                                       const std::string& name)
    : weights_(weights), bias_(bias),
      in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), padding_(padding), stride_(stride),
      name_(name) {
}

std::vector<int> CPUMulticoreConv2D::getOutputShape(const std::vector<int>& input_shape) const {
    int batch = input_shape[0];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    
    int out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    return {batch, out_channels_, out_h, out_w};
}

bool CPUMulticoreConv2D::forward(const float* input, const std::vector<int>& input_shape,
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
    
    int output_size = batch * out_channels_ * out_h * out_w;
    for (int i = 0; i < output_size; i++) {
        int channel = (i / (out_h * out_w)) % out_channels_;
        output[i] = bias_[channel];
    }
    
    for (int b = 0; b < batch; b++) {
        convolve(input + b * in_channels_ * in_h * in_w,
                output + b * out_channels_ * out_h * out_w,
                batch, in_h, in_w, out_h, out_w);
    }
    
    return true;
}

void CPUMulticoreConv2D::convolve(const float* input, float* output,
                                  int batch, int in_h, int in_w, int out_h, int out_w) {
    #pragma omp parallel for collapse(3)
    for (int oc = 0; oc < out_channels_; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                
                for (int ic = 0; ic < in_channels_; ic++) {
                    for (int kh = 0; kh < kernel_size_; kh++) {
                        for (int kw = 0; kw < kernel_size_; kw++) {
                            int ih = oh * stride_ + kh - padding_;
                            int iw = ow * stride_ + kw - padding_;
                            
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int input_idx = ic * in_h * in_w + ih * in_w + iw;
                                int weight_idx = oc * in_channels_ * kernel_size_ * kernel_size_ +
                                               ic * kernel_size_ * kernel_size_ +
                                               kh * kernel_size_ + kw;
                                sum += input[input_idx] * weights_[weight_idx];
                            }
                        }
                    }
                }
                
                int output_idx = oc * out_h * out_w + oh * out_w + ow;
                output[output_idx] += sum;
            }
        }
    }
}

