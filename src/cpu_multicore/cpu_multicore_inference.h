#ifndef CPU_MULTICORE_INFERENCE_H
#define CPU_MULTICORE_INFERENCE_H

#include "../common/model_loader.h"
#include "cpu_multicore_layers.h"
#include <vector>
#include <memory>

/**
 * CPU Multicore Inference Engine using OpenMP
 * Implements CNN inference on CPU using parallel processing
 */
class CPUMulticoreInference {
public:
    CPUMulticoreInference();
    ~CPUMulticoreInference();
    
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

#endif // CPU_MULTICORE_INFERENCE_H

