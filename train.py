#%%
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
from eval_metrics import estimate_loss

# set the device early on
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load configuration
config = ModelConfig.from_yaml("model_config.yaml")

hartokenizer = RegexTokenizer()
if os.path.exists("models/tokenizer.model"):
    hartokenizer.load("models/tokenizer.model")
else:
    # assertion error: model needs to be trained
    raise AssertionError("Model needs to be trained")

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
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
min_loss = float("inf")
save_state = True
load_model = False


# save location
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# save updated model weights
def save_state(model, optimizer, iter, model_dir):
    """
    Save the model and optimizer state.
    """
    save_path = os.path.join(model_dir, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter
    }, save_path)
    print(f"Model saved to {save_path}")

def load_state(model, optimizer, model_dir='models'):
    """
    Load the model and optimizer state.
    """
    # get the latest model
    model_file = model_dir + "/model.pth"
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {model_file}")
#%%
for iter in range(config.max_iters):
    # sample a batch of data
    xb,yb = next(train_data_iter)
    logits, loss = model(xb, yb)
    # backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # loss metrics - poor estimate because uses only one point
    # consider a way this can be done for more
    # do this ocassionally
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        train_loss = estimate_loss(model, train_data_iter, config.eval_iters, device=device)
        val_loss = estimate_loss(model, val_data_iter, config.eval_iters, device=device)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        if val_loss < min_loss and save_state:
            min_loss = val_loss
            save_state(model, optimizer, iter, model_dir)
        
#%%
# generate from model
prompt = "First Citizen:"
print(f"Prompt: {prompt}")
prompt_tokens = hartokenizer.encode(prompt)
# convert to torch 
prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
# generate from model
max_new_tokens = config.max_new_tokens
# %%
# print output
output = model.generate(prompt_tensor, config.max_new_tokens)
output_list = output[0].tolist()
output_text = hartokenizer.decode(output_list)
print(f"Completion: {output_text}")

# %%
