#!/bin/bash

# init.sh - Setup and training script for GPT project
set -e  # Exit on any error
cd $(dirname "$0")  # Change to the directory of the script

echo "Starting GPT project initialization..."

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt &> /dev/null

if [ ! -d "edu_fineweb10b" ]; then
    echo "Downloading data..."
    python download_dataset.py

    if [ -f "fineweb.tar.gz" ]; then
        tar -xzf fineweb.tar.gz
        if [ -d "edu_fineweb10" ]; then
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
    echo "Data folder edu_fineweb10 already exists. Skipping download."
fi

echo "All tasks completed successfully!"
echo "You can now train your model" 