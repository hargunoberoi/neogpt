#%%
from harbpe import RegexTokenizer
from utils import ModelConfig
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT

# Load configuration
config = ModelConfig.from_yaml("model_config.yaml")

with open("input.txt", "r") as f:
    text = f.read()

hartokenizer = RegexTokenizer()
if os.path.exists("models/tokenizer.model"):
    hartokenizer.load("models/tokenizer.model")
else:
    # assertion error: model needs to be trained
    raise AssertionError("Model needs to be trained")
# %% 
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
text_tokens= hartokenizer.encode(text)
n = int(len(text_tokens) * 0.9)
# convert text tokens to tensor
text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)
train_data = text_tokens[:n]
val_data = text_tokens[n:]
vocab_size = len(hartokenizer.vocab)

#%%
model = GPT(vocab_size, config.n_embd, config.n_head, config.block_size, config.dropout)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
#%%
for iter in range(config.max_iters):
    if iter % config.eval_iters == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        train_loss = losses['train'].item()
        val_loss = losses['val'].item()
        # print with two decimals
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
    
    # sample a batch of data
    xb,yb = get_batch('train')

    logits, loss = model(xb, yb)
    # backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
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
