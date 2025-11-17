#include <iostream>
#include <string>
#include <vector>
#include "cpu/cpu_inference.h"
#include "common/timer.h"

/**
 * Main entry point for CNN inference
 * 
 * Usage:
 *   ./cnn_inference --model models/ --image test.jpg --device cpu
 */
int main(int argc, char* argv[]) {
    std::string model_dir = "models";
    std::string image_path;
    std::string device = "cpu";
    
    // Parse command line arguments (simple implementation)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: ./cnn_inference [options]\n"
                      << "Options:\n"
                      << "  --model DIR    Model directory (default: models/)\n"
                      << "  --image FILE   Input image file\n"
                      << "  --device DEV   Device: cpu or gpu (default: cpu)\n"
                      << "  --help         Show this help\n";
            return 0;
        }
    }
    
    std::cout << "=== CNN Inference ===" << std::endl;
    std::cout << "Model directory: " << model_dir << std::endl;
    std::cout << "Device: " << device << std::endl;
    
    if (device == "cpu") {
        CPUInference inference;
        
        if (!inference.initialize(model_dir)) {
            std::cerr << "Failed to initialize CPU inference engine" << std::endl;
            return 1;
        }
        
        // For now, create dummy input (will be replaced with image loader)
        std::vector<float> dummy_input(1 * 3 * 224 * 224, 0.0f);
        std::vector<float> output(1000);
        
        Timer timer;
        timer.start();
        
        if (!inference.infer(dummy_input.data(), output.data())) {
            std::cerr << "Inference failed" << std::endl;
            return 1;
        }
        
        timer.stop();
        
        std::cout << "\nInference completed in " << timer.elapsed_ms() << " ms" << std::endl;
        
        // Find top 5 predictions
        std::vector<std::pair<float, int>> predictions;
        for (int i = 0; i < 1000; i++) {
            predictions.push_back({output[i], i});
        }
        std::sort(predictions.rbegin(), predictions.rend());
        
        std::cout << "\nTop 5 predictions:" << std::endl;
        for (int i = 0; i < 5; i++) {
            std::cout << "  " << (i+1) << ". Class " << predictions[i].second
                      << " (" << predictions[i].first * 100 << "%)" << std::endl;
        }
        
    } else if (device == "gpu") {
        std::cout << "GPU inference not yet implemented" << std::endl;
        return 1;
    } else {
        std::cerr << "Unknown device: " << device << std::endl;
        return 1;
    }
    
    return 0;
}

