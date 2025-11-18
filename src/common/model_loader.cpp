#include "model_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

ModelLoader::ModelLoader() : weights_loaded_(false), total_params_(0) {
}

ModelLoader::~ModelLoader() {
    cleanup();
}

void ModelLoader::cleanup() {
    for (auto& pair : weights_) {
        if (pair.second != nullptr) {
            delete[] pair.second;
        }
    }
    weights_.clear();
    shapes_.clear();
}

float* ModelLoader::loadWeightFile(const std::string& filepath, int size) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return nullptr;
    }
    
    float* weights = new float[size];
    file.read(reinterpret_cast<char*>(weights), size * sizeof(float));
    
    if (static_cast<size_t>(file.gcount()) != size * sizeof(float)) {
        std::cerr << "Error: File size mismatch for " << filepath << std::endl;
        delete[] weights;
        return nullptr;
    }
    
    file.close();
    return weights;
}

bool ModelLoader::loadWeights(const std::string& model_dir) {
    cleanup();
    
    std::vector<std::pair<std::string, int>> weight_files = {
        {"conv0.weight", 64 * 3 * 3 * 3},
        {"conv0.bias", 64},
        {"conv1.weight", 128 * 64 * 3 * 3},
        {"conv1.bias", 128},
        {"conv2.weight", 256 * 128 * 3 * 3},
        {"conv2.bias", 256},
        {"conv3.weight", 256 * 256 * 3 * 3},
        {"conv3.bias", 256},
        {"conv4.weight", 512 * 256 * 3 * 3},
        {"conv4.bias", 512},
        {"conv5.weight", 512 * 512 * 3 * 3},
        {"conv5.bias", 512},
        {"conv6.weight", 512 * 512 * 3 * 3},
        {"conv6.bias", 512},
        {"conv7.weight", 512 * 512 * 3 * 3},
        {"conv7.bias", 512},
        {"fc0.weight", 4096 * 25088},
        {"fc0.bias", 4096},
        {"fc1.weight", 4096 * 4096},
        {"fc1.bias", 4096},
        {"fc2.weight", 1000 * 4096},
        {"fc2.bias", 1000}
    };
    
    std::cout << "Loading model weights from: " << model_dir << std::endl;
    
    for (const auto& file_info : weight_files) {
        std::string filename = file_info.first;
        int size = file_info.second;
        
        std::string filepath = model_dir + "/" + filename + ".dat";
        float* weights = loadWeightFile(filepath, size);
        
        if (weights == nullptr) {
            std::cerr << "Failed to load: " << filename << std::endl;
            cleanup();
            return false;
        }
        
        weights_[filename] = weights;
        total_params_ += size;
        
        std::vector<int> shape;
        if (filename.find("conv") != std::string::npos && filename.find("weight") != std::string::npos) {
            if (filename == "conv0.weight") shape = {64, 3, 3, 3};
            else if (filename == "conv1.weight") shape = {128, 64, 3, 3};
            else if (filename == "conv2.weight") shape = {256, 128, 3, 3};
            else if (filename == "conv3.weight") shape = {256, 256, 3, 3};
            else if (filename == "conv4.weight") shape = {512, 256, 3, 3};
            else if (filename == "conv5.weight") shape = {512, 512, 3, 3};
            else if (filename == "conv6.weight") shape = {512, 512, 3, 3};
            else if (filename == "conv7.weight") shape = {512, 512, 3, 3};
        } else if (filename.find("conv") != std::string::npos && filename.find("bias") != std::string::npos) {
            shape = {size};
        } else if (filename.find("fc") != std::string::npos && filename.find("weight") != std::string::npos) {
            if (filename == "fc0.weight") shape = {4096, 25088};
            else if (filename == "fc1.weight") shape = {4096, 4096};
            else if (filename == "fc2.weight") shape = {1000, 4096};
        } else if (filename.find("fc") != std::string::npos && filename.find("bias") != std::string::npos) {
            shape = {size};
        }
        
        shapes_[filename] = shape;
        std::cout << "  [OK] " << filename << " - " << size << " parameters" << std::endl;
    }
    
    weights_loaded_ = true;
    std::cout << "\n[OK] Model loaded successfully!" << std::endl;
    std::cout << "Total parameters: " << total_params_ << std::endl;
    
    return true;
}

float* ModelLoader::getWeights(const std::string& layer_name) {
    auto it = weights_.find(layer_name);
    if (it != weights_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<int> ModelLoader::getShape(const std::string& layer_name) {
    auto it = shapes_.find(layer_name);
    if (it != shapes_.end()) {
        return it->second;
    }
    return {};
}

