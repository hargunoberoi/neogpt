#%% import modules/libraries
from data import  ShardDataset, ShardIterableDataset
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import logging
import torch
import tiktoken
from itertools import cycle
from model import GPT, GPTConfig
import argparse
from utils import save_state, load_state, get_lr
import wandb
import time

# create required directories
model_dir = "models"
logs_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# set config
parser = argparse.ArgumentParser(description="Train a GPT model")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
# take argument for logging level
parser.add_argument("--log_level", type=str, default="DEBUG", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
# take bool argument to restart training
parser.add_argument("--restart", action='store_true', help="Restart training from the last checkpoint")
# get a runid for wandb
parser.add_argument("--run_id", type=str, default="30452", help="Run ID for wandb")
parser.add_argument("--logging", action='store_true', help="Enable wandb logging")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

improved_vocab_size = 50304
config = GPTConfig(vocab_size=improved_vocab_size)
enc = tiktoken.get_encoding("gpt2")
# get the dataset and dataloader

val_dataset = ShardDataset("edu_fineweb10b", config.block_size, split="val")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
eval_iters = 1000
max_iters = 200000 # default set to 5 epochs
# gpt based updates
total_batch_size = 2**19 #524288 ~0.5M as per gpt paper
B = args.batch_size
T = config.block_size
assert total_batch_size % (B * T) == 0, "This will be true because B and T are both powers of 2"

#%% Train model

def train(rank,world_size):
    start_iter = 0
    #%% define stuff
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3045' 
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # logic of the training loop
    train_iter_ds = ShardIterableDataset(
        data_dir="edu_fineweb10b",
        block_size=config.block_size,
        split="train",
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_iter_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset,
                                             batch_size=args.batch_size, 
                                             shuffle=False,
                                             drop_last=True)
    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)
                                        
    torch.set_float32_matmul_precision('high')
    # get the model
    model = GPT(config).to(device)  # create model and move to device
    raw_model = model
    optimizer = raw_model.configure_optimizers(weight_decay=1e-1, learning_rate=max_lr, device=device)
    # load checkpoint
    if args.restart:
        # check if model file exists
        if not os.path.exists("models/model.pth"):
            logging.error("No checkpoint found to restart training. Exiting.")
            return
        else:
            logging.info("Loading checkpoint to restart training...")
            start_iter = load_state(model, optimizer, model_dir=model_dir)
    model = torch.compile(model) 
    model = DDP(model, device_ids=[rank]) # ddp wrapper for distributed training
    grad_accum_steps = total_batch_size // (B*T* world_size)  # gradient accumulation steps per process
    skip_batches = start_iter * grad_accum_steps
    for _ in range(skip_batches):
        next(train_iter) 
    #%% training loop
    run = None
    if rank == 0:
        logging.info(f"Starting training with {world_size} processes.")
        if args.logging:
            run_id = args.run_id if args.run_id else wandb.util.generate_id()
            run = wandb.init(
                project="neogpt",
                config=config.__dict__,
                name="fineweb-training",
                id = run_id,
                resume=True,
                settings = wandb.Settings(init_timeout=120),
            )
            # alert that a run has started
            run.alert(
                title="Run started",
                text=f"Run {run.name} with id {run.id} has started.",
                level="INFO"
            )
    for iter in range(start_iter, max_iters):
        # sample a batch of data
        t0 = time.time()
        last_step = iter == max_iters - 1
        # once in a while evaluate on validation loss
        if iter % eval_iters == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    xb, yb = next(val_iter)
                    xb, yb = xb.to(device), yb.to(device)
                    # with torch.autocast(device_type=device.type, dtype=torch.bfloat16): 
                    _, loss = model(xb, yb)
                    loss = loss / val_loss_steps  # average loss over steps
                    val_loss_accum += loss.detach()
                val_loss_tensor = torch.tensor(val_loss_accum, device=device)  # create a tensor for loss accumulation
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)  # average loss across all processes
                val_loss_accum = val_loss_tensor.item()  # convert back to Python float
                if rank == 0:
                    logging.info(f"Validation loss at step {iter}: {val_loss_accum:.4f}")
                    if run is not None:
                        run.log({"val/loss": val_loss_accum}, step=iter)
                        run.alert(
                        title="Validation",
                        text="Validation loss at step {}: {:.4f}".format(iter, val_loss_accum),
                        level="INFO"
                        )
                        with open(os.path.join(logs_dir, "val_loss.txt"), "a") as f:
                            f.write(f"Step {iter}: {val_loss_accum:.4f}\n")

        model.train()  # switch back to training mode
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            xb, yb = next(train_iter)
            xb, yb = xb.to(device), yb.to(device)
            model.require_backward_sync = (micro_step == grad_accum_steps - 1)  # only sync gradients on the last micro step
            # with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            loss.backward()  # accumulate gradients
        loss_tensor = torch.tensor(loss_accum, device=device)  # create a tensor for loss accumulation
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)  # average loss across all processes
        loss_accum = loss_tensor.item()  # convert back to Python float
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        # set the new learning rate as per scheduler
        lr = get_lr(iter, warmup_steps, max_lr, min_lr, max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()  # synchronize to ensure all operations are complete
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps * world_size  # total tokens processed in this step
        if rank == 0:
            if args.logging and run is not None:
                run.log({"train/loss": loss_accum, "train/lr": lr, "train/norm": norm, "tokens/sec": tokens_processed / dt}, step=iter)
                logging.info(f"Training loss at step {iter}: {loss_accum:.4f}")
                with open(os.path.join(logs_dir, "train_loss.txt"), "a") as f:
                    f.write(f"Step {iter}: {loss_accum:.4f}\n")
                if (iter > 0 and iter % eval_iters == 0) or last_step:
                    save_state(iter, raw_model, optimizer, model_dir= model_dir)
                    model_artifact = wandb.Artifact(f"neogpt",type="model")
                    model_artifact.add_file(os.path.join(model_dir, "model.pth"))
                    # add training and validation loss files as well
                    model_artifact.add_file(os.path.join(logs_dir, "train_loss.txt"))
                    model_artifact.add_file(os.path.join(logs_dir, "val_loss.txt"))
                    run.log_artifact(model_artifact,aliases=["latest"])
            else:
                logging.info(f"Training loss at step {iter}: {loss_accum:.4f}")
                if (iter > 0 and iter % eval_iters == 0) or last_step:
                    save_state(iter, raw_model, optimizer, model_dir= model_dir)


    dist.destroy_process_group()  # clean up any existing process group
# %%
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train,args=(world_size,),nprocs=world_size)


if __name__ == "__main__":
    main()