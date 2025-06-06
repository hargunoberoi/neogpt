#!/bin/bash

# init.sh - Setup and training script for GPT project
set -e  # Exit on any error
cd $(dirname "$0")  # Change to the directory of the script

echo "Starting GPT project initialization..."

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt &> /dev/null

# wandb login
if ! command -v wandb &> /dev/null; then
    echo "wandb is not installed. Installing wandb..."
    pip install wandb
fi
echo "Logging in to Weights & Biases..."
wandb login --relogin

if [ ! -d "edu_fineweb10b" ]; then
    echo "Downloading data..."
    python download_dataset.py

    if [ -f "fineweb.tar.gz" ]; then
        tar -xzf fineweb.tar.gz
        if [ -d "edu_fineweb10b" ]; then
            rm fineweb.tar.gz
        else
            echo "Extraction failed. Directory not found after untar."
            exit 1
        fi
    else
        echo "Download failed. fineweb.tar.gz not found."
        exit 1
    fi
else
    echo "Data folder edu_fineweb10b already exists. Skipping download."
fi

# download the model using download_models.py 
# checking if the model directory exists
if [ ! -d "models" ]; then
    echo "Downloading model..."
    if python download_models.py; then
        echo "Download succeeded."
    else
        echo "Download failed! Start training from scratch" >&2
        # clean up partial directory if it was created by download_models.py
        rm -rf models
        exit 1
    fi
else
    echo "Model directory already exists. Skipping download."
fi
echo "All tasks completed successfully!"