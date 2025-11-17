# Setup Guide

## Repository Setup

This repository contains the source code for GPU-accelerated CNN inference implementation.

### Cloning on CIMS Server

```bash
# Clone repository
git clone <repository-url>
cd Project

# Install Python dependencies (for model export)
pip install -r requirements.txt

# Export model weights (if not already in repository)
python export_pretrained_weights.py

# Build project
make cpu

# Run inference
./build/cnn_inference_cpu --model models/ --device cpu
```

### Directory Structure

```
.
├── src/                    # Source code
│   ├── common/            # Shared utilities
│   ├── cpu/               # CPU implementations
│   ├── gpu/               # GPU implementations (TODO)
│   └── layers/            # Layer base classes
├── models/                 # Model weights (must be generated locally)
├── data/                   # Dataset (download separately)
├── Makefile               # Build configuration
└── README.md              # Project documentation
```

### Important Notes

- **Model weights** (`models/*.dat`): Large files (~500MB). Generate locally using `export_pretrained_weights.py`
- **Dataset** (`data/food-101/`): Large files (~5GB). Download separately from Kaggle
- Source code and scripts are version controlled

### Generating Model Weights

If model weights are not in the repository, generate them:

```bash
python export_pretrained_weights.py --output models
```

### Downloading Dataset

Download Food-101 dataset from:
- https://www.kaggle.com/datasets/kmader/food41

Extract to `data/food-101/` directory.

