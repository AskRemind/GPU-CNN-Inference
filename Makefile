# Makefile for CNN Inference Project

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -Wall -march=native
NVCCFLAGS = -std=c++17 -O3 -arch=sm_75 --use_fast_math -Xcompiler -fPIC
LDFLAGS = 
OPENMP_FLAGS = -fopenmp
CUDA_LDFLAGS = -lcudart

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = -I$(SRC_DIR)

# Source files
COMMON_SRC = $(SRC_DIR)/common/model_loader.cpp
CPU_SRC = $(SRC_DIR)/cpu/cpu_conv2d.cpp $(SRC_DIR)/cpu/cpu_layers.cpp $(SRC_DIR)/cpu/cpu_inference.cpp
CPU_MULTICORE_SRC = $(SRC_DIR)/cpu_multicore/cpu_multicore_conv2d.cpp $(SRC_DIR)/cpu_multicore/cpu_multicore_layers.cpp $(SRC_DIR)/cpu_multicore/cpu_multicore_inference.cpp
GPU_CU_SRC = $(SRC_DIR)/gpu/gpu_conv2d.cu $(SRC_DIR)/gpu/gpu_layers.cu $(SRC_DIR)/gpu/gpu_inference.cu
MAIN_SRC = $(SRC_DIR)/main.cpp

# Object files
COMMON_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_SRC))
CPU_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPU_SRC))
CPU_MULTICORE_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPU_MULTICORE_SRC))
GPU_OBJ = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(GPU_CU_SRC))
MAIN_OBJ = $(BUILD_DIR)/main.o

# Targets
TARGET_CPU = $(BUILD_DIR)/cnn_inference_cpu
TARGET_CPU_MULTICORE = $(BUILD_DIR)/cnn_inference_cpu_multicore
TARGET_GPU = $(BUILD_DIR)/cnn_inference_gpu

.PHONY: all clean cpu cpu_multicore gpu

all: cpu cpu_multicore gpu

cpu: $(TARGET_CPU)

cpu_multicore: $(TARGET_CPU_MULTICORE)

gpu: $(TARGET_GPU)

$(TARGET_CPU): $(COMMON_OBJ) $(CPU_OBJ) $(MAIN_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $@"

$(TARGET_CPU_MULTICORE): $(COMMON_OBJ) $(CPU_MULTICORE_OBJ) $(MAIN_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(LDFLAGS) $(OPENMP_FLAGS)
	@echo "Build complete: $@"

$(TARGET_GPU): $(COMMON_OBJ) $(GPU_OBJ) $(MAIN_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
	@echo "Build complete: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_DIR) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

# Test runs
test: cpu
	./$(TARGET_CPU) --model models/ --device cpu

test_multicore: cpu_multicore
	./$(TARGET_CPU_MULTICORE) --model models/ --device cpu_multicore

test_gpu: gpu
	./$(TARGET_GPU) --model models/ --device gpu

