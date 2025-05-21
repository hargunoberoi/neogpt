
#%%
from harbpe import RegexTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT

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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
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

   
T = block_size = 8
B = batch_size = 32
V = vocab_size = len(hartokenizer.vocab)
E = n_embd = 64
max_iters = 3000
#eval_interval = 500
learning_rate = 3e-4
eval_iters = 100
n_head = 6
n_layer = 6
dropout = 0.2
#%%
model = GPT(vocab_size, n_embd, n_head, block_size, dropout)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#%%
for iter in range(max_iters):
    if iter % eval_iters == 0 or iter == max_iters - 1:
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
max_new_tokens = 200


# %%
# print output
output = model.generate(prompt_tensor, max_new_tokens)
output_list = output[0].tolist()
output_text = hartokenizer.decode(output_list)

print(f"Completion: {output_text}")

# %%
