#%%
"""
Simplified inference script for GPT model.
Loads trained model weights and runs inference on input prompts.
"""
import tiktoken
from utils import generate_from_model, load_state
import os
import torch
from model import GPT, GPTConfig
# Set device and print it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
#%%
prompt = "Hargun Singh Oberoi is"
print(f"Prompt: {prompt}")

config = GPTConfig(vocab_size=50304)# Use the improved vocab size)
enc = tiktoken.get_encoding("gpt2")  # Use GPT-2 encoding for tokenization
# 1. Create a model from scratch and generate
print("\n[From Scratch] Generating with randomly initialized model...")

scratch_model = GPT(config).to(device)
scratch_model.eval()
# load model weights from models/model.pth
# if os.path.exists("models/model.pth"):
#     model_file = torch.load("models/model.pth", map_location=device,weights_only=True)
#     try:
#         scratch_model.load_state_dict(model_file['model_state_dict'])
#     except Exception as e:
#         print(f"Error loading model state di  ct: {e}")
output_text = generate_from_model(prompt, scratch_model, enc, device, max_new_tokens=20)
print(f"[From Scratch Completion]: {output_text}")
#%%
# 2. Load pretrained GPT-2 and generate
print("\n[Pretrained GPT-2] Generating with GPT-2 weights...")
#%%
pretrained_model = GPT.from_pretrained("gpt2").to(device)
pretrained_model.eval()
output_text = generate_from_model(prompt, pretrained_model, enc, device, max_new_tokens=20)
print(f"[Pretrained GPT-2 Completion]: {output_text}")

#%%