#%% import modules/libraries
import os
from itertools import cycle
import wandb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from model import GPT
from data import  StreamingTextDataset
from utils import ModelConfig
from utils import save_state, load_state, estimate_loss, generate_from_model
from harbpe import RegexTokenizer
 #%% set up device, config and tokenizer, wandb
config = ModelConfig.from_yaml("model_config.yaml")

hartokenizer = RegexTokenizer()
if os.path.exists("models/tokenizer.model"):
    hartokenizer.load("models/tokenizer.model")
else:
    # assertion error: model needs to be trained
    raise AssertionError("Model needs to be trained")

def train(rank, world_size):
    model_dir = 'models'
    save_model = True
    min_loss = 100
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank,world_size=world_size)

    torch.cuda.set_device(rank)
    train_dataset = StreamingTextDataset(config.block_size, hartokenizer,rank=rank, world_size=world_size)
        # get the dataloader 
    train_data = DataLoader(train_dataset, batch_size=config.batch_size,shuffle=False)
    train_data_iter = cycle(train_data)

    sample_prompts = [
                "Hargun Singh Oberoi is",
            ]
    generation_table = None
    # set wandb init
    if rank == 0:
        run_id = '4722794' + wandb.util.generate_id()
        run = wandb.init(
            project="neogpt",
            config=config.__dict__,
            name="fineweb-training",
            id = run_id,
        )

        # alert that a run has started
        run.alert(
            title="Run started",
            text=f"Run {run.name} with id {run.id} has started.",
            level="INFO"
        )
        # generate samples
        
        
        generation_table = wandb.Table(columns = ["iter"] + sample_prompts)

    model = GPT(config.vocab_size, config.n_embd, config.n_head, config.n_layer, config.block_size, config.dropout)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=config.learning_rate)
    
    for iter in range(config.max_iters):
        # sample a batch of data
        xb, yb = next(train_data_iter)
        xb, yb = xb.to(rank), yb.to(rank)
        logits, loss = ddp_model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step() 

        if rank == 0 and (iter % config.eval_interval == 0 or iter == config.max_iters - 1):
            train_loss = estimate_loss(ddp_model.module, train_data_iter, config.eval_iters, device=rank)
            print(f'Rank {rank}, Iter {iter}, Train Loss: {train_loss.item()}')

            generated_samples = [generate_from_model(prompt, ddp_model.module, hartokenizer, config, rank) for prompt in sample_prompts]
            
            # log to wandb
            generation_table.add_data(iter, *generated_samples)
            metric_dict = {
                "train_loss": train_loss.item(),
                "iter": iter,
                "generations": wandb.Table(data=generation_table.data, columns=generation_table.columns)
            }
            run.log(metric_dict)
            if (((min_loss - train_loss) / min_loss) > config.update_threshold) and save_model:
                    min_loss = train_loss
                    save_state(ddp_model.module, optimizer, model_dir)
                    # save model weights as latest
                    model_artifact = wandb.Artifact(f"model_{run.id}", type="model")
                    model_artifact.add_file(os.path.join(model_dir, "model.pth"))
                    run.log_artifact(model_artifact,aliases=["latest"])
    if rank == 0:
        run.finish()
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)

if __name__ == '__main__':
    main()