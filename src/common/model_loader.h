#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

/**
 * ModelLoader: Loads VGG11 weights from binary .dat files
 * 
 * Weight files are stored as float32 binary arrays that can be
 * directly loaded into memory for inference.
 */
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();
    
    /**
     * Load all model weights from directory
     * @param model_dir Directory containing weight files (*.dat)
     * @return true if successful
     */
    bool loadWeights(const std::string& model_dir);
    
    /**
     * Load a single weight file
     * @param filepath Path to .dat file
     * @param size Expected number of elements
     * @return Pointer to loaded weights (must be freed), nullptr on error
     */
    static float* loadWeightFile(const std::string& filepath, int size);
    
    /**
     * Get weight array for a layer
     * @param layer_name e.g., "conv0.weight", "fc0.bias"
     * @return Pointer to weight array, nullptr if not found
     */
    float* getWeights(const std::string& layer_name);
    
    /**
     * Get shape of a weight array
     * @param layer_name Layer name
     * @return Shape vector, empty if not found
     */
    std::vector<int> getShape(const std::string& layer_name);
    
    /**
     * Check if weights are loaded
     */
    bool isLoaded() const { return weights_loaded_; }
    
    /**
     * Get total number of parameters
     */
    size_t getTotalParams() const { return total_params_; }
    
private:
    std::unordered_map<std::string, float*> weights_;
    std::unordered_map<std::string, std::vector<int>> shapes_;
    bool weights_loaded_;
    size_t total_params_;
    
    void cleanup();
};

#endif // MODEL_LOADER_H

