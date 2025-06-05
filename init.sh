#!/bin/bash

# init.sh - Setup and training script for GPT project
set -e  # Exit on any error
cd $(dirname "$0")  # Change to the directory of the script

echo "Starting GPT project initialization..."

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt &> /dev/null

# Download data if folder doesn't exist
# 1. check if the folder edu_fineweb10 exists
if [ ! -d "edu_fineweb10" ]; then
    echo "Downloading data..."
    python download_data.py &> /dev/null
    # untar the file
    tar -xzf fineweb.tar.gz
    # clean up after
    if [ -d "edu_fineweb10" ]; then
        rm fineweb.tar.gz
    else
        echo "Data download failed. Please check the download_data.py script."
        exit 1
    fi
fi
echo "All tasks completed successfully!"
echo "You can now train your model" 