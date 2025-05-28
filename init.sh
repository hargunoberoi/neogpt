#!/bin/bash

# init.sh - Setup and training script for GPT project
set -e  # Exit on any error
cd $(dirname "$0")  # Change to the directory of the script

echo "Starting GPT project initialization..."

# Clone the harbpe library from GitHub if not already present
if [ ! -d "harbpe" ]; then
    echo "Cloning harbpe tokenizer library..."
    git clone https://github.com/hargunoberoi/harbpe
else
    echo "harbpe tokenizer library already present, skipping clone."
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check if models folder exists
if [ ! -d "models" ]; then
    echo "Models folder not found. ..."
    mkdir -p models
    python download_models.py
# else check if tokenizer.model does not exist, if not run download_models.py
else 
    if [ ! -f "models/tokenizer.model" ]; then
        echo "tokenizer.model not found in models folder. Downloading models."
        python download_models.py
    else
        echo "Models already downloaded, skipping download step."
    fi      
fi
    
# Run main training
echo "Starting main model training..."
python train.py

echo "All tasks completed successfully!"
echo "Your GPT model training pipeline has finished." 