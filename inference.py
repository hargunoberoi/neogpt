"""
Simplified inference script for GPT model.
Loads trained model weights and runs inference on input prompts.
"""
import tiktoken
from utils import generate_from_model
import os
import torch
from model import GPT, GPTConfig
# Set device and print it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

prompt = "Hargun Singh Oberoi is"
print(f"Prompt: {prompt}")

config = GPTConfig()
enc = tiktoken.get_encoding("gpt2")  # Use GPT-2 encoding for tokenization
# 1. Create a model from scratch and generate
print("\n[From Scratch] Generating with randomly initialized model...")
from model import GPTConfig
scratch_config = GPTConfig(
    vocab_size=config.vocab_size,
    n_embd=config.n_embd,
    n_head=config.n_head,
    n_layer=config.n_layer,
    block_size=config.block_size
)
scratch_model = GPT(scratch_config).to(device)
scratch_model.eval()
output_text = generate_from_model(prompt, scratch_model, enc, device, max_new_tokens=100)
print(f"[From Scratch Completion]: {output_text}")

# 2. Load pretrained GPT-2 and generate
print("\n[Pretrained GPT-2] Generating with GPT-2 weights...")
pretrained_model = GPT.from_pretrained("gpt2").to(device)
pretrained_model.eval()
output_text = generate_from_model(prompt, pretrained_model, enc, device, max_new_tokens=100)
print(f"[Pretrained GPT-2 Completion]: {output_text}")
