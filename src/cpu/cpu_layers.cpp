#include "cpu_layers.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <iostream>

bool CPUMaxPool2D::forward(const float* input, const std::vector<int>& input_shape,
                          float* output, std::vector<int>& output_shape) {
    int batch = input_shape[0];
    int channels = input_shape[1];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    int out_h = in_h / stride_;
    int out_w = in_w / stride_;
    
    output_shape = {batch, channels, out_h, out_w};
    
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float max_val = -std::numeric_limits<float>::max();
                    
                    for (int kh = 0; kh < kernel_size_; kh++) {
                        for (int kw = 0; kw < kernel_size_; kw++) {
                            int ih = oh * stride_ + kh;
                            int iw = ow * stride_ + kw;
                            
                            if (ih < in_h && iw < in_w) {
                                int idx = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                                max_val = std::max(max_val, input[idx]);
                            }
                        }
                    }
                    
                    int out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
    
    return true;
}

CPULinear::CPULinear(const float* weights, const float* bias,
                     int in_features, int out_features,
                     const std::string& name)
    : weights_(weights), bias_(bias),
      in_features_(in_features), out_features_(out_features),
      name_(name) {
}

bool CPULinear::forward(const float* input, const std::vector<int>& input_shape,
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
    
    for (int b = 0; b < batch; b++) {
        for (int of = 0; of < out_features_; of++) {
            float sum = bias_[of];
            
            for (int inf = 0; inf < in_features_; inf++) {
                int input_idx = b * in_features_ + inf;
                int weight_idx = of * in_features_ + inf;
                sum += input[input_idx] * weights_[weight_idx];
            }
            
            output[b * out_features_ + of] = sum;
        }
    }
    
    return true;
}

bool CPUSoftmax::forward(const float* input, const std::vector<int>& input_shape,
                        float* output, std::vector<int>& output_shape) {
    output_shape = input_shape;
    
    int batch = input_shape[0];
    int num_classes = 1;
    for (size_t i = 1; i < input_shape.size(); i++) {
        num_classes *= input_shape[i];
    }
    
    for (int b = 0; b < batch; b++) {
        float max_val = input[b * num_classes];
        for (int i = 1; i < num_classes; i++) {
            max_val = std::max(max_val, input[b * num_classes + i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float exp_val = std::exp(input[b * num_classes + i] - max_val);
            output[b * num_classes + i] = exp_val;
            sum += exp_val;
        }
        
        for (int i = 0; i < num_classes; i++) {
            output[b * num_classes + i] /= sum;
        }
    }
    
    return true;
}

