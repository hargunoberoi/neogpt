#%% import modules/libraries
from utils import ModelConfig
from data import TextDataset, StreamingTextDataset
from torch.utils.data import DataLoader
import os
import sys
import torch
import tiktoken
from itertools import cycle
from model import GPT, GPTConfig
import argparse
from utils import save_state, load_state, estimate_loss, generate_from_model
import wandb
import time

# set config
parser = argparse.ArgumentParser(description="Train a GPT model")
parser.add_argument("--max_iters", type=int, default=5, help="Maximum number of training iterations")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
tokenizer = tiktoken.get_encoding("gpt2")

#%% set train and validation data loaders
# input data and tokenize 
with open("input.txt", "r") as f:
    raw_text = f.read()
tokens = tokenizer.encode(raw_text)
split_idx = int(0.9 * len(tokens))

train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]


# %% Initialize model 
# Load configuration
"""
All improvements based on AK suggestions
- reduce matmul precision
- set "good" vocab size
- use torch compile
- use flash attention
"""
torch.set_float32_matmul_precision('high')
improved_vocab_size = 50304
config = GPTConfig(vocab_size=improved_vocab_size)
# get the dataset and dataloader
train_dataset = TextDataset(train_tokens, config.block_size)
val_dataset = TextDataset(val_tokens, config.block_size)
train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
train_data_iter = cycle(train_data)
# get the model
model = GPT(config)
model = model.to(device)
model = torch.compile(model)  # compile the model for performance

learning_rate = 6e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#%% Train model
for iter in range(args.max_iters):
    # sample a batch of data
    t0 = time.time()
    xb, yb = next(train_data_iter)
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    # backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()  # synchronize to ensure all operations are complete
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = args.batch_size * config.block_size
    print(f"step {iter} | loss {loss.item():.4f} | dt {dt:.2f}s | tokens/sec {tokens_processed / dt:.2f}")

# %%
