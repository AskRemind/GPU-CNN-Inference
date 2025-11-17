#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split Food-101 dataset into subset with selected classes
"""

import os
import sys
from pathlib import Path
import shutil

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Selected classes (10-15 categories for testing)
SELECTED_CLASSES = [
    'apple_pie',
    'bread_pudding',
    'cheesecake',
    'chicken_curry',
    'chocolate_cake',
    'donuts',
    'french_fries',
    'hamburger',
    'ice_cream',
    'pizza',
    'sushi',
    'tacos',
    'tiramisu',
    'waffles'
]

# Paths
BASE_DIR = Path("data/food-101")
SUBSET_DIR = Path("data/food-101-subset")
META_DIR = BASE_DIR / "meta"
IMAGES_DIR = BASE_DIR / "images"

def split_dataset():
    """Split dataset by selected classes"""
    
    # Create subset directories
    subset_images = SUBSET_DIR / "images"
    subset_meta = SUBSET_DIR / "meta"
    subset_images.mkdir(parents=True, exist_ok=True)
    subset_meta.mkdir(parents=True, exist_ok=True)
    
    # Read original train/test splits
    with open(META_DIR / "train.txt", 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]
    
    with open(META_DIR / "test.txt", 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    # Filter by selected classes
    selected_train = [f for f in train_files if f.split('/')[0] in SELECTED_CLASSES]
    selected_test = [f for f in test_files if f.split('/')[0] in SELECTED_CLASSES]
    
    # Copy images and create train/test splits
    print(f"Processing {len(SELECTED_CLASSES)} classes...")
    print(f"Train samples: {len(selected_train)}")
    print(f"Test samples: {len(selected_test)}")
    
    # Copy images
    for class_name in SELECTED_CLASSES:
        src_dir = IMAGES_DIR / class_name
        dst_dir = subset_images / class_name
        
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            print(f"[OK] Copied {class_name}")
        else:
            print(f"[X] {class_name} not found")
    
    # Write train.txt
    with open(subset_meta / "train.txt", 'w') as f:
        for item in selected_train:
            f.write(f"{item}\n")
    
    # Write test.txt
    with open(subset_meta / "test.txt", 'w') as f:
        for item in selected_test:
            f.write(f"{item}\n")
    
    # Write classes.txt
    with open(subset_meta / "classes.txt", 'w') as f:
        for cls in SELECTED_CLASSES:
            f.write(f"{cls}\n")
    
    # Create class mapping (index to name)
    class_mapping = {i: cls for i, cls in enumerate(SELECTED_CLASSES)}
    
    print("\n" + "="*50)
    print("Dataset split complete!")
    print("="*50)
    print(f"Subset location: {SUBSET_DIR}")
    print(f"Classes: {len(SELECTED_CLASSES)}")
    print(f"Train samples: {len(selected_train)}")
    print(f"Test samples: {len(selected_test)}")
    print("\nClass mapping:")
    for idx, cls in class_mapping.items():
        print(f"  {idx}: {cls}")
    
    return SUBSET_DIR

if __name__ == "__main__":
    split_dataset()

