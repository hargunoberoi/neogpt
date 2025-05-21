"""
Bigram language model implemenation with custom tokenizer"""

# open file input.txt
with open("input.txt", "r") as f:
    text = f.read()
#%%
from harbpe import RegexTokenizer
import os

#%%

hartokenizer = RegexTokenizer()
if os.path.exists("models/tokenizer.model"):
    hartokenizer.load("models/tokenizer.model")
else:
    # assertion error: model needs to be trained
    raise AssertionError("Model needs to be trained")
# %% tokenize input 

text_tokens= hartokenizer.encode(text)
# %% Create counts matrix for bigram model

import torch
V = vocab_size = len(hartokenizer.vocab)

W = torch.zeros((V, V), dtype=torch.float32)
# %%
# loop through the text_tokens and append the count at the 
# i+1 element

for token1, token2 in zip(text_tokens[:-1], text_tokens[1:]):
    W[token1, token2] += 1


# %%

# add a small factor to get non-zero probabilities at any given column
correction_count = 1
W_corrected = W + correction_count

P = W_corrected / W_corrected.sum(axis=1, keepdim=True)

# %%

# Inference

import torch.nn.functional as F

prompt = "First Citizen:"
print(f"Prompt: {prompt}")
prompt_tokens = hartokenizer.encode(prompt)
tokens = len(prompt_tokens)
max_tokens = 200

while tokens < max_tokens:
    # get the last token
    last_token = prompt_tokens[-1]
    # get the probabilities of the next token
    probs = P[last_token].detach()
    # sample from the distribution
    next_token = torch.multinomial(probs, num_samples=1)
    prompt_tokens.append(next_token.item())
    tokens+=1

decoded_output = hartokenizer.decode(prompt_tokens)
print(f"Completion: {decoded_output}")
# %% Neural network style implementation
T = block_size = 8
B = batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
# split text_tokens
n = int(len(text_tokens) * 0.9)
# convert text tokens to tensor
text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)
train_data = text_tokens[:n]
val_data = text_tokens[n:]
eval_iters = 200
max_iters = 3000
learning_rate = 1e-3



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
#%%
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None

        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self,idx,max_new_tokens):
        
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']}, val loss {losses['val']}")
    
    # sample a batch of data
    xb,yb = get_batch('train')

    logits, loss = model(xb, yb)
    # backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

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
