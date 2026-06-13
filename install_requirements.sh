#!/bin/bash

# MatSKRAFT Environment Setup Script
# Tested on Ampere GPUs (CUDA 12.4), Python 3.9

# Initialize conda for script context
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create and activate environment
conda create -n matskraft python=3.9 -y
conda activate matskraft

# Install PyTorch 2.4.0 with CUDA 12.4
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install DGL 2.4.0 with CUDA 12.4 and PyTorch 2.4
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# Install torch-scatter for PyTorch 2.4.0 + CUDA 12.4
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Install remaining dependencies
pip install -r requirements.txt

echo "MatSKRAFT environment setup complete."
