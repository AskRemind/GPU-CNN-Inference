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
     * Run inference on a batch of images
     * @param image_data Image data (NCHW format, normalized), where N is batch_size
     * @param output Output probabilities for each class (N x num_classes)
     * @param batch_size Number of images in the batch
     * @return true if successful
     */
    bool infer(const float* image_data, float* output, int batch_size = 1);
    
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

