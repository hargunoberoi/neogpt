#%% import modules/libraries
from harbpe import RegexTokenizer
from utils import ModelConfig
from data import TextDataset
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
    name="colab-training",
    id = id,
)
#%% set train and validation data loaders
# input data and tokenize 
with open("input.txt", "r") as f:
    raw_text = f.read()
tokens = hartokenizer.encode(raw_text)
split_idx = int(0.9 * len(tokens))

train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

train_dataset = TextDataset(train_tokens, config.block_size)
val_dataset = TextDataset(val_tokens, config.block_size)

# get the dataloader 
train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# get infinite iterator using cycle
train_data_iter = cycle(train_data)
val_data_iter = cycle(val_data)
# %% Initialize model 
model = GPT(config.vocab_size, config.n_embd, config.n_head, config.block_size, config.dropout)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
min_loss = 100.0 # setting a very high initial loss
save_model = True
load_model = False

#%%generate from model
def generate_from_model(prompt, model, tokenizer, config, device):
    print(f"Prompt: {prompt}")
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    output = model.generate(prompt_tensor, config.max_new_tokens)
    output_list = output[0].tolist()
    output_text = tokenizer.decode(output_list)
    print(f"Completion: {output_text}")
    return output_text

sample_prompt = "Thou shall not"
out = generate_from_model(sample_prompt, model, hartokenizer, config, device)
print("Initial Generation before training")
print(f"Sample output: {out}")
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
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        train_loss = estimate_loss(model, train_data_iter, config.eval_iters, device=device)
        val_loss = estimate_loss(model, val_data_iter, config.eval_iters, device=device)
        out = generate_from_model(sample_prompt, model, hartokenizer, config, device)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        # make a clean line to separate outputs
        print("-" * 50)
        # print output
        print(f"Sample output: {out}")
        print("-" * 50)
        metric_dict = {
            "train_loss": train_loss.mean().item(),
            "val_loss": val_loss.mean().item(),
            "iter": iter,
            "sample_output": out,
        }
        run.log(metric_dict)
        if (((min_loss - val_loss) / min_loss) > config.training.update_threshold) and save_model:
            min_loss = val_loss
            save_state(model, optimizer, iter, model_dir)
            # save model weights as latest
            model_artifact = wandb.Artifact("model",type="model")
            model_artifact.add_file(os.path.join(model_dir, "model.pth"))
            run.log_artifact(model_artifact,aliases=["latest"])

run.finish()

# %%
