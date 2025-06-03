#%% import modules/libraries
from utils import ModelConfig
from data import TextDataset, StreamingTextDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import sys
import torch
import tiktoken
from itertools import cycle
from model import GPT, GPTConfig
import argparse
from utils import save_state, load_state, estimate_loss, generate_from_model, get_lr
import wandb
import time

# set config
parser = argparse.ArgumentParser(description="Train a GPT model")
parser.add_argument("--max_iters", type=int, default=5, help="Maximum number of training iterations")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

improved_vocab_size = 50304
config = GPTConfig(vocab_size=improved_vocab_size)
# get the dataset and dataloader
train_dataset = TextDataset(train_tokens, config.block_size)
val_dataset = TextDataset(val_tokens, config.block_size)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_iters = args.max_iters

# gpt based updates
total_batch_size = 2**19 #524288 ~0.5M as per gpt paper
B = args.batch_size
T = config.block_size
assert total_batch_size % (B * T) == 0, "This will be true because B and T are both powers of 2"


#%% Train model


def train(rank,world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3045' 
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # logic of the training loop
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, 
                                           batch_size=args.batch_size, 
                                           sampler=sampler,
                                           drop_last=True)
    train_iter = cycle(train_loader)
                                        
    # jump to here
    torch.set_float32_matmul_precision('high')
    # get the model
    model = GPT(config)
    model = model.to(rank)
    model = torch.compile(model)
    model = DDP(model, device_ids=[rank]) # ddp wrapper for distributed training
    raw_model = model.module  # get the raw model for saving state
    optimizer = raw_model.configure_optimizers(weight_decay=1e-1, learning_rate=max_lr, device=device)
    grad_accum_steps = total_batch_size // (B*T* world_size)  # gradient accumulation steps per process
    # training loop
    for iter in range(max_iters):
        # sample a batch of data
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            xb, yb = next(train_iter)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            model.require_backward_sync = (micro_step == grad_accum_steps - 1)  # only sync gradients on the last micro step
            loss.backward()  # accumulate gradients
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # average loss across all processes
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        # set the new learning rate as per scheduler
        lr = get_lr(iter, warmup_steps, max_lr, min_lr, max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()  # synchronize to ensure all operations are complete
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = args.batch_size * config.block_size * grad_accum_steps * world_size  # total tokens processed in this step
        if rank == 0:
            print(f"step {iter} | loss {loss_accum:.4f} | lr {lr:.3e} | norm: {norm:.4f} | dt {dt:.2f}s | tokens/sec {tokens_processed / dt:.2f}")

    dist.destroy_process_group()  # clean up any existing process group
# %%
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train,args=(world_size,),nprocs=world_size)


if __name__ == "__main__":
    main()