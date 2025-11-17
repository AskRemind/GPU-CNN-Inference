#!/usr/bin/env python3
"""
Export PyTorch VGG11 model weights to binary format for C++ inference.

This script loads a pre-trained VGG11 model from torchvision, extracts the weights,
and saves them as binary .dat files that can be loaded by the C++ implementation.

Requirements:
    pip install torch torchvision

Usage:
    python3 export_weights.py --output_dir models/

Author: GPU-CNN-Inference Project
"""

import argparse
import os
import json
import numpy as np
import torch
import torchvision.models as models


def export_conv_layer(module, name_prefix, output_dir):
    """Export convolutional layer weights and bias."""
    # Export weights: [out_channels, in_channels, kernel_h, kernel_w]
    weight = module.weight.data.cpu().numpy()
    weight_file = os.path.join(output_dir, f"{name_prefix}.weight.dat")
    weight.astype(np.float32).tofile(weight_file)
    
    # Export bias: [out_channels]
    bias = module.bias.data.cpu().numpy()
    bias_file = os.path.join(output_dir, f"{name_prefix}.bias.dat")
    bias.astype(np.float32).tofile(bias_file)
    
    weight_size = weight.size
    bias_size = bias.size
    
    print(f"  [OK] {name_prefix}.weight - {weight_size} parameters")
    print(f"  [OK] {name_prefix}.bias - {bias_size} parameters")
    
    return weight_size + bias_size


def export_linear_layer(module, name_prefix, output_dir):
    """Export fully connected layer weights and bias."""
    # Export weights: [out_features, in_features]
    weight = module.weight.data.cpu().numpy()
    weight_file = os.path.join(output_dir, f"{name_prefix}.weight.dat")
    weight.astype(np.float32).tofile(weight_file)
    
    # Export bias: [out_features]
    bias = module.bias.data.cpu().numpy()
    bias_file = os.path.join(output_dir, f"{name_prefix}.bias.dat")
    bias.astype(np.float32).tofile(bias_file)
    
    weight_size = weight.size
    bias_size = bias.size
    
    print(f"  [OK] {name_prefix}.weight - {weight_size} parameters")
    print(f"  [OK] {name_prefix}.bias - {bias_size} parameters")
    
    return weight_size + bias_size


def export_model_info(output_dir):
    """Export model architecture information to JSON."""
    model_info = {
        "architecture": "VGG11",
        "pretrained": "ImageNet",
        "input_size": [224, 224, 3],
        "num_classes": 1000,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "layers": [
            {"type": "conv2d", "name": "conv0", "index": 0, "in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "relu", "name": "relu_0"},
            {"type": "maxpool2d", "name": "pool0", "kernel_size": 2, "stride": 2, "padding": 0},
            {"type": "conv2d", "name": "conv1", "index": 1, "in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "relu", "name": "relu_1"},
            {"type": "maxpool2d", "name": "pool1", "kernel_size": 2, "stride": 2, "padding": 0},
            {"type": "conv2d", "name": "conv2", "index": 2, "in_channels": 128, "out_channels": 256, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "relu", "name": "relu_2"},
            {"type": "conv2d", "name": "conv3", "index": 3, "in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "relu", "name": "relu_3"},
            {"type": "maxpool2d", "name": "pool2", "kernel_size": 2, "stride": 2, "padding": 0},
            {"type": "conv2d", "name": "conv4", "index": 4, "in_channels": 256, "out_channels": 512, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "relu", "name": "relu_4"},
            {"type": "conv2d", "name": "conv5", "index": 5, "in_channels": 512, "out_channels": 512, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "relu", "name": "relu_5"},
            {"type": "maxpool2d", "name": "pool3", "kernel_size": 2, "stride": 2, "padding": 0},
            {"type": "conv2d", "name": "conv6", "index": 6, "in_channels": 512, "out_channels": 512, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "relu", "name": "relu_6"},
            {"type": "conv2d", "name": "conv7", "index": 7, "in_channels": 512, "out_channels": 512, "kernel_size": 3, "padding": 1, "stride": 1},
            {"type": "relu", "name": "relu_7"},
            {"type": "maxpool2d", "name": "pool4", "kernel_size": 2, "stride": 2, "padding": 0},
            {"type": "linear", "name": "fc0", "index": 0, "in_features": 25088, "out_features": 4096},
            {"type": "relu", "name": "relu_8"},
            {"type": "linear", "name": "fc1", "index": 1, "in_features": 4096, "out_features": 4096},
            {"type": "relu", "name": "relu_9"},
            {"type": "linear", "name": "fc2", "index": 2, "in_features": 4096, "out_features": 1000}
        ]
    }
    
    info_file = os.path.join(output_dir, "model_info.json")
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"  [OK] model_info.json - Model architecture information")


def export_imagenet_classes(output_dir):
    """Export ImageNet class names."""
    # Create a placeholder file (users can download the full list if needed)
    classes_file = os.path.join(output_dir, "imagenet_classes.txt")
    
    # Note: Full ImageNet class list is 1000 classes
    # For a complete list, download from: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    with open(classes_file, 'w') as f:
        f.write("# ImageNet 1000 classes\n")
        f.write("# For full list, see: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt\n")
        f.write("# This file is a placeholder. Replace with full class list if needed.\n")
    
    print(f"  [OK] imagenet_classes.txt - Class names placeholder")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch VGG11 weights to binary format')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for weight files (default: models/)')
    parser.add_argument('--model_name', type=str, default='vgg11',
                        choices=['vgg11', 'vgg11_bn'],
                        help='Model variant (default: vgg11)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Exporting VGG11 Model Weights ===")
    print(f"Output directory: {args.output_dir}")
    print("")
    
    # Load pre-trained VGG11 model
    print("Loading pre-trained VGG11 model from torchvision...")
    model = models.vgg11(weights='IMAGENET1K_V1')
    model.eval()
    
    print("Extracting weights...")
    print("")
    
    total_params = 0
    
    # Extract features (convolutional layers)
    features = model.features
    
    # Conv blocks: 0, 3, 7, 10, 14, 17, 21, 24
    conv_indices = [0, 3, 7, 10, 14, 17, 21, 24]
    for i, idx in enumerate(conv_indices):
        conv_layer = features[idx]
        params = export_conv_layer(conv_layer, f"conv{i}", args.output_dir)
        total_params += params
    
    # Extract classifier (fully connected layers)
    classifier = model.classifier
    
    # FC layers: 0, 3, 6
    fc_indices = [0, 3, 6]
    for i, idx in enumerate(fc_indices):
        fc_layer = classifier[idx]
        params = export_linear_layer(fc_layer, f"fc{i}", args.output_dir)
        total_params += params
    
    # Export model info
    export_model_info(args.output_dir)
    
    # Export class names
    export_imagenet_classes(args.output_dir)
    
    print("")
    print("[OK] Model weights exported successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"")
    print(f"Weight files saved to: {args.output_dir}/")
    print("")
    print("Note: imagenet_classes.txt is a placeholder.")
    print("      Download full ImageNet class list from:")
    print("      https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")


if __name__ == '__main__':
    main()

