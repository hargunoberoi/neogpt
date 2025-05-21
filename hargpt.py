"""
Transformer model with the following items: 
- positional + token embeddings
- attention blocks
- linear layer
"""
#%%
from harbpe import RegexTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyper parameters
batch_size = 32
block_size = 8

# %% Neural network style implementation
T = block_size = 8
B = batch_size = 32
max_iters = 3000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
E = n_embd = 64
n_head = 6
dropout = 0.2
#%%

with open("input.txt", "r") as f:
    text = f.read()

hartokenizer = RegexTokenizer()
if os.path.exists("models/tokenizer.model"):
    hartokenizer.load("models/tokenizer.model")
else:
    # assertion error: model needs to be trained
    raise AssertionError("Model needs to be trained")
# %% tokenize input 

text_tokens= hartokenizer.encode(text)
n = int(len(text_tokens) * 0.9)
# convert text tokens to tensor
text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)
train_data = text_tokens[:n]
val_data = text_tokens[n:]

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

class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Head(nn.Module):

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):

        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        uniform_attn = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        uniform_attn = uniform_attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        softmax_attn = F.softmax(uniform_attn, dim=-1)
        softmax_attn = self.dropout(softmax_attn)
        out = softmax_attn @ v
        return out

class GPT(nn.Module):

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
