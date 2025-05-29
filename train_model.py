#%% import modules/libraries
from harbpe import RegexTokenizer
from utils import ModelConfig
from data import TextDataset, StreamingTextDataset
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT
from itertools import cycle
from utils import save_state, load_state, estimate_loss
import wandb
 #%% set up device, config and tokenizer, wandb
# set the device early on
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = "models"
# Load configuration
config = ModelConfig.from_yaml("model_config.yaml")
#%%
hartokenizer = RegexTokenizer()
if os.path.exists("models/tokenizer.model"):
    hartokenizer.load("models/tokenizer.model")
else:
    # assertion error: model needs to be trained
    raise AssertionError("Model needs to be trained")

# generate hexadecimal code for id

id = '4722794' + wandb.util.generate_id()

run = wandb.init(
    project="neogpt",
    config=config.__dict__,
    name="fineweb-training",
    id = id,
)

# alert that a run has started
run.alert(
    title="Run started",
    text=f"Run {run.name} with id {run.id} has started.",
    level="INFO"
)
#%% set train and validation data loaders
# input data and tokenize 

train_dataset = StreamingTextDataset(config.block_size, hartokenizer)

# get the dataloader 
train_data = DataLoader(train_dataset, batch_size=config.batch_size )
train_data_iter = cycle(train_data)

# %% Initialize model 
model = GPT(config.vocab_size, config.n_embd, config.n_head, config.n_layer, config.block_size, config.dropout)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
min_loss = 100.0 # setting a very high initial loss
save_model = True
load_model = False

#%%generate from model
def generate_from_model(prompt, model, tokenizer, config, device):
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    output = model.generate(prompt_tensor, config.max_new_tokens)
    output_list = output[0].tolist()
    output_text = tokenizer.decode(output_list)
    return output_text

sample_prompts = [
    "Hargun Singh Oberoi is",
]

generation_table = wandb.Table(columns = ["iter"] + sample_prompts)
#%% Train model
for iter in range(config.max_iters):
    # sample a batch of data
    xb, yb = next(train_data_iter)
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    # backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # do this ocassionally to save the model state
    # if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
    #     train_loss = estimate_loss(model, train_data_iter, config.eval_iters, device=device)
    #     # randomly select a sample prompt
    #     outs = [generate_from_model(prompt, model, hartokenizer, config, device) for prompt in sample_prompts]
    #     # log to generation table
    #     generation_table.add_data(iter, *outs)
    #     print(f"step {iter}: train loss {train_loss:.4f}")
    #     metric_dict = {
    #         "train_loss": train_loss.item(),
    #         "iter": iter,
    #         "generations": wandb.Table(data=generation_table.data, columns=generation_table.columns)
    #     }
    #     run.log(metric_dict)
    #     if (((min_loss - train_loss) / min_loss) > config.update_threshold) and save_model:
    #         min_loss = train_loss
    #         save_state(model, optimizer, model_dir)
    #         # save model weights as latest
    #         model_artifact = wandb.Artifact(f"model_{run.id}", type="model")
    #         model_artifact.add_file(os.path.join(model_dir, "model.pth"))
    #         run.log_artifact(model_artifact,aliases=["latest"])

run.finish()

# %%
