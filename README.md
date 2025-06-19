## NeoGPT: Recreating GPT-2 124M from Scratch

This project aims to recreate the 124M parameter GPT-2 model using custom PyTorch code.
All of this code is imitating Andrej Karpathy's nanogpt project.
The goal is to understand and document every step of the process, from data loading and sharding to distributed training and checkpointing, without relying on high-level black-box libraries.

### Project Structure & File Overview

- **model.py**: Defines the GPT model architecture and configuration, closely following the original GPT-2 124M design (transformer blocks, attention, etc.).
- **train_ddp.py**: Main distributed training script using PyTorch DDP. Handles multi-GPU training, checkpointing, and optional experiment logging (wandb).
- **train_ddp_lite.py**: Variant of the training script using a custom lightweight data loader (Andrej Karpathy implementation) for efficient sharded data access.
- **train_gpt2.py**: Copy of Andrej Karpathy implementation for testing and comparision.
- **data.py**: Implements custom dataset and iterable dataset classes for sharded data loading, as well as `DataLoaderLite` for fast, memory-efficient batch access.
- **utils.py**: Utility functions for saving/loading model checkpoints and learning rate scheduling.
- **download_dataset.py, extract_dataset.py, download_models.py**: Scripts for preparing and managing datasets and pretrained weights.
- **inference.py**: Script for running inference/generation with a trained model.
- **scripts/**: Contains additional scripts for data/tokenizer preparation, configuration, and experiment management.

### Workflow Summary
1. **Prepare Data**: Use the scripts in `scripts/` and `data.py` to download, shard, and preprocess your dataset into the `edu_fineweb10b/` directory.
2. **Configure Model**: Adjust hyperparameters and model config in `model.py` and the training scripts as needed.
3. **Train**: Launch distributed training with `train_ddp.py` or `train_ddp_lite.py` for efficient multi-GPU training. Checkpoints and logs are saved in `models/` and `wandb/`.
4. **Evaluate/Infer**: Use `inference.py` to generate text or evaluate the trained model.
