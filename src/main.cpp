#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "common/timer.h"

#ifdef BUILD_CPU
#include "cpu/cpu_inference.h"
#endif

#ifdef BUILD_CPU_MULTICORE
#include "cpu_multicore/cpu_multicore_inference.h"
#endif

#ifdef BUILD_GPU
#include "gpu/gpu_inference.h"
#endif

/**
 * Main entry point for CNN inference
 * 
 * Usage:
 *   ./cnn_inference --model models/ --image test.jpg --device cpu|cpu_multicore|gpu
 */
int main(int argc, char* argv[]) {
    std::string model_dir = "models";
    std::string image_path;
    std::string device = "cpu";
    int batch_size = 1;
    
    // Parse command line arguments (simple implementation)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
            if (batch_size <= 0) {
                std::cerr << "Error: Invalid batch size: " << batch_size << std::endl;
                return 1;
            }
        } else if (arg == "--help") {
            std::cout << "Usage: ./cnn_inference [options]\n"
                      << "Options:\n"
                      << "  --model DIR    Model directory (default: models/)\n"
                      << "  --image FILE   Input image file\n"
                      << "  --device DEV   Device: cpu, cpu_multicore, or gpu (default: cpu)\n"
                      << "  --batch_size N Batch size (default: 1)\n"
                      << "  --help         Show this help\n";
            return 0;
        }
    }
    
    std::cout << "=== CNN Inference ===" << std::endl;
    std::cout << "Model directory: " << model_dir << std::endl;
    std::cout << "Device: " << device << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    
    // For now, create dummy input (will be replaced with image loader)
    std::vector<float> dummy_input(batch_size * 3 * 224 * 224, 0.0f);
    std::vector<float> output(batch_size * 1000);
    
    Timer timer;
    timer.start();
    
    if (device == "cpu") {
#ifdef BUILD_CPU
        CPUInference inference;
        
        if (!inference.initialize(model_dir)) {
            std::cerr << "Failed to initialize CPU inference engine" << std::endl;
            return 1;
        }
        
        if (!inference.infer(dummy_input.data(), output.data(), batch_size)) {
            std::cerr << "Inference failed" << std::endl;
            return 1;
        }
#else
        std::cerr << "CPU implementation not compiled in this build" << std::endl;
        return 1;
#endif
        
    } else if (device == "cpu_multicore") {
#ifdef BUILD_CPU_MULTICORE
        CPUMulticoreInference inference;
        
        if (!inference.initialize(model_dir)) {
            std::cerr << "Failed to initialize CPU Multicore inference engine" << std::endl;
            return 1;
        }
        
        if (!inference.infer(dummy_input.data(), output.data(), batch_size)) {
            std::cerr << "Inference failed" << std::endl;
            return 1;
        }
#else
        std::cerr << "CPU Multicore implementation not compiled in this build" << std::endl;
        return 1;
#endif
        
    } else if (device == "gpu") {
#ifdef BUILD_GPU
        GPUInference inference;
        
        if (!inference.initialize(model_dir)) {
            std::cerr << "Failed to initialize GPU inference engine" << std::endl;
            return 1;
        }
        
        if (!inference.infer(dummy_input.data(), output.data(), batch_size)) {
            std::cerr << "Inference failed" << std::endl;
            return 1;
        }
#else
        std::cerr << "GPU implementation not compiled in this build" << std::endl;
        return 1;
#endif
        
    } else {
        std::cerr << "Unknown device: " << device << std::endl;
        std::cerr << "Available devices: cpu, cpu_multicore, gpu" << std::endl;
        return 1;
    }
    
    timer.stop();
    
    std::cout << "\nInference completed in " << timer.elapsed_ms() << " ms" << std::endl;
    
    // Find top 5 predictions (for first image in batch if batch_size > 1)
    if (batch_size == 1) {
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
    } else {
        std::cout << "\nBatch inference completed (" << batch_size << " images)" << std::endl;
    }
    
    return 0;
}

