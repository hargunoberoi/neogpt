#%% import modules/libraries
from utils import ModelConfig, get_lr
from data import ShardDataset
from torch.utils.data import DataLoader
import os
import torch
from itertools import cycle
from model import GPT, GPTConfig
import argparse
import time

# set config
parser = argparse.ArgumentParser(description="Train a GPT model")
parser.add_argument("--max_iters", type=int, default=5, help="Maximum number of training iterations")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use ShardDataset instead of TextDataset
improved_vocab_size = 50304
config = GPTConfig(vocab_size=improved_vocab_size)
train_dataset = ShardDataset("edu_fineweb10b", config.block_size, split="train")
val_dataset = ShardDataset("edu_fineweb10b", config.block_size, split="val")
train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
train_data_iter = cycle(train_data)

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
model = GPT(config)
model = model.to(device)
model = torch.compile(model) if device.type == 'cuda' else model  # compile if on CUDA

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_iters = args.max_iters
optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=max_lr, device=device)

# gpt based updates
total_batch_size = 2**19 #524288 ~0.5M as per gpt paper
B = args.batch_size
T = config.block_size
assert total_batch_size % (B * T) == 0, "This will be true because B and T are both powers of 2"
grad_accum_steps = total_batch_size // (B*T)

#%% Train model
for iter in range(max_iters):
    t0 = time.time()
    # Periodically run validation, similar to train_ddp.py (every 100 iters)
    if iter % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            val_data_iter = cycle(val_data)
            for _ in range(val_loss_steps):
                xb, yb = next(val_data_iter)
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                loss = loss / val_loss_steps
                val_loss_accum += loss.item()
            print(f"Validation loss at step {iter}: {val_loss_accum:.4f}")
        model.train()

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        xb, yb = next(train_data_iter)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        loss = loss / grad_accum_steps
        loss_accum += loss.item()
        loss.backward()  # accumulate gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
    # set the new learning rate as per scheduler
    lr = get_lr(iter, warmup_steps, max_lr, min_lr, max_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()  # synchronize to ensure all operations are complete
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = B * T * grad_accum_steps
    print(f"step {iter} | loss {loss_accum:.4f} | lr {lr:.3e} | norm: {norm:.4f} | dt {dt:.2f}s | tokens processed {tokens_processed} | tokens/sec {tokens_processed / dt:.2f}")

# %%
