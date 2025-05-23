#!/bin/bash

# init.sh - Setup and training script for GPT project
set -e  # Exit on any error

echo "Starting GPT project initialization..."

# get the harbpe library from github
git clone https://github.com/hargunoberoi/harbpe

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check if models folder exists
if [ ! -d "models" ]; then
    echo "Models folder not found. Creating and running tokenizer training..."
    mkdir -p models
fi
    
# Check if tokenizer.model exists
if [ ! -f "models/tokenizer.model" ]; then
    echo "tokenizer.model not found. Running tokenizer training..."
    python train_tokenizer.py

    # Verify tokenizer training created expected files
    if [ -f "models/tokenizer.model" ] || [ -f "models/tokenizer.vocab" ]; then
        echo "Tokenizer files created successfully"
    else
        echo "Warning: Expected tokenizer files not found"
        exit 1
    fi
else
    echo "tokenizer.model already exists, skipping tokenizer training"
fi

# Run main training
echo "Starting main model training..."
python train.py

echo "All tasks completed successfully!"
echo "Your GPT model training pipeline has finished." 