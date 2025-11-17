#ifndef CPU_INFERENCE_H
#define CPU_INFERENCE_H

#include "../common/model_loader.h"
#include "cpu_layers.h"
#include <vector>
#include <memory>

/**
 * CPU Sequential Inference Engine
 * Implements CNN inference on CPU using sequential processing
 */
class CPUInference {
public:
    CPUInference();
    ~CPUInference();
    
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

#endif // CPU_INFERENCE_H

