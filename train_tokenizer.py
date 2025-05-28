#%%
# open file input.txt
#%%
from harbpe import RegexTokenizer
import os
from utils import ModelConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT
from utils import save_state, load_state, estimate_loss
import wandb 

with open("input.txt", "r") as f:
    text = f.read()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%% # load model_config.yaml and get vocab_size 
config = ModelConfig.from_yaml("model_config.yaml")
#%% train tokenizer
hartokenizer = RegexTokenizer(max_tokens=config.vocab_size)
# check if models/tokenizer.model does not exist,
if os.path.exists("models/tokenizer.model"):
    print("Tokenizer already exists")
else:
    hartokenizer.train(text)
    os.makedirs("models", exist_ok=True)
    prefix = os.path.join("models", "tokenizer")
    hartokenizer.save(prefix)

# %% Randomly initialize the GPT model if model doesn't exist
if os.path.exists("models/model.pth"):
    print("Model already exists")
else:
    model = GPT(config.vocab_size, config.n_embd, config.n_head, config.block_size, config.dropout)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # save model weights
    save_state(model, optimizer, 0, "models")

# %% create tokenizer_artifact for project neogpt
# first create tokenizer_artifact called tokenizer
upload_tokenizer = False
upload_untrained_model = True

if not upload_tokenizer:
    print("Skipping tokenizer upload")
else:
    run = wandb.init(project="neogpt")
    tokenizer_artifact = wandb.Artifact("tokenizer", type="tokenizer")
    tokenizer_artifact.add_file("models/tokenizer.model")
    run.log_artifact(tokenizer_artifact)
    run.finish()

if not upload_untrained_model:
    print("Skipping untrained model upload")
else:
    run = wandb.init(project="neogpt", reinit=True)
    model_artifact = wandb.Artifact("model", type="model")
    model_artifact.add_file("models/model.pth")
    run.log_artifact(model_artifact)
    run.finish()
