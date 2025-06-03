#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
import tiktoken
import inspect
#%%x
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
        # peform initialization scaling according to gpt-2 paper
        self.c_proj.HARSCALE_INIT = 1 
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

        out = F.scaled_dot_product_attention(q,k,v,is_causal=True) # flash attention
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
        self.c_proj.HARSCALE_INIT = 1  # perform initialization scaling according to gpt-2 paper

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
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'HARSCALE_INIT'):
                std *= (2 * self.config.n_layer)**-0.5 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads pretrained GPT-2 model weights from huggingface 
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        print("Loading pretrained GPT-2 model weights from huggingface")

        # create config dict
        options = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        config_args = options[model_type]
        config_args["block_size"] = 1024
        config_args["vocab_size"] = 50257  # GPT-2 vocab size
        
        config = GPTConfig(**config_args)
        # initialize model
        model = cls(config)

        # get weights through state_dict
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard mask buffer, not needed
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]


        # get hugging face model
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()
        # get model weights
        # avoid huggingface model keys that are not weights
        # list transposed weights, they need to be taken care of separately
        hf_sd_keys = hf_sd.keys()
        hf_sd_keys = [k for k in hf_sd_keys if not k.endswith('.attn.bias')]  # discard mask buffer
        hf_sd_keys = [k for k in hf_sd_keys if not k.endswith('.attn.masked_bias')]  # discard masked bias 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(hf_sd_keys), f"State dict keys mismatch: {len(sd_keys)} vs {len(hf_sd_keys)}"
        for k in hf_sd_keys:
            # special treatment weights
            if any(k.endswith(t) for t in transposed): 
                assert hf_sd[k].shape[::-1] == sd[k].shape, f"Shape mismatch for transposed key {k}: {hf_sd[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    # transpose the weights
                    sd[k].copy_(hf_sd[k].T)
            else:        # normal weights simple copy
                assert hf_sd[k].shape == sd[k].shape, f"Shape mismatch for key {k}: {hf_sd[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    # copy the weights
                    sd[k].copy_(hf_sd[k])
        return model  # return the model

    def forward(self,idx,targets=None):
        B,T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0,T,dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # B,T,C
        tok_emb = self.transformer.wte(idx)  # B,T,C
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() > 1]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, 
                                      lr=learning_rate,
                                      betas=(0.9, 0.95),
                                      eps=1e-8,
                                      fused=use_fused)
        return optimizer

    def generate(self, idx,max_new_tokens):
        block_size = self.config.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            # sample
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
# if this file is run, try a sample prompt

if __name__ == "__main__":
    config = GPTConfig()
    model = GPT.from_pretrained("gpt2")  # load pretrained GPT-2 model
    model.eval()  # set to eval mode
    sample_prompt = "SolidGoldMagikarp"
    enc = tiktoken.get_encoding("gpt2")
    num_return_sequences = 5
    tokens = enc.encode(sample_prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # repeat for multiple sequences
    out = model.generate(tokens, max_new_tokens=50)
    # get in plain text
    out_text = [enc.decode(out[i].tolist()) for i in range(num_return_sequences)]
    for i, text in enumerate(out_text):
        print(f"Generated text {i+1}: {text}")
        print("="*50)