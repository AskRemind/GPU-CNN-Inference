# Makefile for CNN Inference Project

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -Wall -march=native
NVCCFLAGS = -std=c++17 -O3 -arch=sm_75 --use_fast_math
LDFLAGS = 

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = -I$(SRC_DIR)

# Source files
COMMON_SRC = $(SRC_DIR)/common/model_loader.cpp
CPU_SRC = $(SRC_DIR)/cpu/cpu_conv2d.cpp $(SRC_DIR)/cpu/cpu_layers.cpp $(SRC_DIR)/cpu/cpu_inference.cpp
MAIN_SRC = $(SRC_DIR)/main.cpp

# Object files
COMMON_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(COMMON_SRC))
CPU_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPU_SRC))
MAIN_OBJ = $(BUILD_DIR)/main.o

# Targets
TARGET_CPU = $(BUILD_DIR)/cnn_inference_cpu

.PHONY: all clean cpu gpu

all: cpu

cpu: $(TARGET_CPU)

$(TARGET_CPU): $(COMMON_OBJ) $(CPU_OBJ) $(MAIN_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIR) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

# Test run
test: cpu
	./$(TARGET_CPU) --model models/ --device cpu

