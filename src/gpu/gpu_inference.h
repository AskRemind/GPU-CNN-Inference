#ifndef GPU_INFERENCE_H
#define GPU_INFERENCE_H

#include "../common/model_loader.h"
#include "gpu_layers.h"
#include <vector>
#include <memory>

/**
 * GPU Inference Engine using CUDA
 * Implements CNN inference on GPU using CUDA kernels
 */
class GPUInference {
public:
    GPUInference();
    ~GPUInference();
    
    /**
     * Initialize inference engine with model weights
     * @param model_dir Directory containing weight files
     * @return true if successful
     */
    bool initialize(const std::string& model_dir);
    
    /**
     * Run inference on a single image
     * @param image_data Image data (CHW format, normalized)
     * @param output Output probabilities for each class
     * @return true if successful
     */
    bool infer(const float* image_data, float* output);
    
    /**
     * Get number of output classes
     */
    int getNumClasses() const { return 1000; }  // ImageNet classes
    
private:
    ModelLoader model_loader_;
    std::vector<std::unique_ptr<Layer>> layers_;
    bool initialized_;
    
    void buildNetwork();
};

#endif // GPU_INFERENCE_H

