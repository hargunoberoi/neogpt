#!/bin/bash

# init.sh - Setup and training script for GPT project
set -e  # Exit on any error
cd $(dirname "$0")  # Change to the directory of the script

echo "Starting GPT project initialization..."

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt &> /dev/null


echo "All tasks completed successfully!"
echo "You can now train your model" 