#ifndef LAYER_BASE_H
#define LAYER_BASE_H

#include <vector>
#include <string>

/**
 * Base class for all CNN layers
 * Provides common interface for CPU and GPU implementations
 */
class Layer {
public:
    virtual ~Layer() = default;
    
    /**
     * Forward pass through the layer
     * @param input Input tensor (flattened or in CHW format)
     * @param input_shape [batch, channels, height, width] or [batch, features]
     * @param output Output tensor (allocated by caller)
     * @param output_shape Output shape (filled by function)
     * @return true if successful
     */
    virtual bool forward(const float* input, const std::vector<int>& input_shape,
                        float* output, std::vector<int>& output_shape) = 0;
    
    /**
     * Get output shape given input shape
     * @param input_shape Input shape
     * @return Output shape
     */
    virtual std::vector<int> getOutputShape(const std::vector<int>& input_shape) const = 0;
    
    /**
     * Get layer name
     */
    virtual std::string getName() const = 0;
};

#endif // LAYER_BASE_H

