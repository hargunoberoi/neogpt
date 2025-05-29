import torch
import torch.nn as nn
import torch.nn.functional as F
   
class AttentionHead(nn.Module):

    def __init__(self, n_embd, num_heads,block_size,dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd,bias=False)
        self.query = nn.Linear(n_embd, n_embd,bias=False)
        self.value = nn.Linear(n_embd, n_embd,bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x) # B,T,E
        k = self.key(x)
        v = self.value(x)

        # done: operations to perform
        # 1. reshape last dimension to N,E (N -> num heads)
        # 2. transpose -3 and -2 to go from B,T,N,H to B,N,T,H
        q = q.view(B, T, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.num_heads, -1).transpose(1, 2)

        uniform_attn = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        uniform_attn = uniform_attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        softmax_attn = F.softmax(uniform_attn, dim=-1)
        softmax_attn = self.dropout(softmax_attn)
        out = softmax_attn @ v # B,N,T,H

        # reverse operations
        out = out.transpose(1, 2) 
        out = out.reshape(B, T, -1)

        # apply projection and dropout
        out = self.dropout(self.proj(out))
        return out 

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
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        self.sa = AttentionHead(n_embd, n_head, block_size, dropout )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # save block size
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        # tokens to embeddings
        input_emb = tok_emb + pos_emb
        # embeddings to attention
        x = self.blocks(input_emb)
        x = self.ln_f(x)
        # attention to logits
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self,idx,max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx