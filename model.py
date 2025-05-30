import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class AttentionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))
        # self.dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B,T,C = x.shape
        qkv = self.c_attn(x) # B,T,3*C
        q,k,v = qkv.split(self.n_embd, dim=-1) # B,T,C
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)

        uniform_attn = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        uniform_attn = uniform_attn.masked_fill(self.bias[:,:,:T, :T] == 0, float('-inf'))
        softmax_attn = F.softmax(uniform_attn, dim=-1)
        # softmax_attn = self.dropout(softmax_attn) # no dropouts in actual implementation
        out = softmax_attn @ v # B,N,T,H

        # reverse operations
        out = out.transpose(1, 2).contiguous() # B,T,N,H 
        out = out.view(B, T, C) # B,T,C

        # apply projection and dropout
        # out = self.dropout(self.proj(out))
        out = self.c_proj(out) # B,T,C
        return out 

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)  # B,T,4*C
        x = self.gelu(x)  # B,T,4*C 
        x = self.c_proj(x)  # B,T,C
        return x  # B,T,C
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = AttentionHead(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls,model_type):
        """
        Loading pretrained GPT-2 model weights from huggingface
        """
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        print(f'loading pretrained model weights from huggingface for {model_type}')

        # n_layer, n_head, n_embd from model type

        config_args_dict = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }
        config_args = config_args_dict[model_type]
        config_args['vocab_size'] = 50257  # GPT-2 vocab size
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        model_weights = model.state_dict()
        model_weights_keys = model_weights.keys()
        model_weights_keys = [k for k in model_weights_keys if not k.endswith('.attn.bias')]

        # init hugging face transformer
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        model_weights_hf = model_hf.state_dict()
        model_weights_hf_keys = model_weights_hf.keys()
        
        # copy while ensuring all parameters are aligned
        # ignore .attn.bias
        model_weights_hf_keys = [k for k in model_weights_hf_keys if not k.endswith('.attn.masked_bias')]
        model_weights_hf_keys = [k for k in model_weights_hf_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(model_weights_hf_keys) == len(model_weights_keys), f"Mismatched keys: {len(model_weights_hf_keys)} vs {len(model_weights_keys)}"
        for key in model_weights_hf_keys:
            if any(key.endswith(t) for t in transposed):
                assert model_weights_hf[key].shape[::-1] == model_weights[key].shape, f"Shape mismatch for {key}: {model_weights_hf[key].shape[::-1]} vs {model_weights[key].shape}"
                with torch.no_grad():
                    model_weights[key].copy_(model_weights_hf[key].t())
            else:
                assert model_weights_hf[key].shape == model_weights[key].shape, f"Shape mismatch for {key}: {model_weights_hf[key].shape} vs {model_weights[key].shape}"
                with torch.no_grad():
                    # check if loop reaches here
                    model_weights[key].copy_(model_weights_hf[key])

        return model
    
    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

if __name__ == "__main__":
    # Example usage
    model = GPT.from_pretrained('gpt2')
    print("Model loaded successfully with config:", model.config)
    # print model weights 
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    # old implemenation
    # def forward(self, idx, targets=None):
    #     B,T = idx.shape
    #     tok_emb = self.token_embedding_table(idx)
    #     pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
    #     # tokens to embeddings
    #     input_emb = tok_emb + pos_emb
    #     # embeddings to attention
    #     x = self.blocks(input_emb)
    #     x = self.ln_f(x)
    #     # attention to logits
    #     logits = self.lm_head(x)

    #     if targets is None:
    #         loss = None
    #     else:
    #         B, T, C = logits.shape
    #         logits = logits.view(B*T, C)
    #         targets = targets.view(B*T)
    #         loss = F.cross_entropy(logits, targets)
    #     return logits, loss

    # def generate(self,idx,max_new_tokens):
        
    #     for _ in range(max_new_tokens):
    #         idx_cond = idx[:, -self.block_size:]
    #         logits, loss = self(idx_cond)
    #         logits = logits[:, -1, :]
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         idx = torch.cat((idx, idx_next), dim=1)
        
    #     return idx

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device) # move to device

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
