"""
Simplified inference script for GPT model.
Loads trained model weights and runs inference on input prompts.
"""
import os
import torch
from harbpe import RegexTokenizer
from utils import ModelConfig
from model import GPT

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load configuration
config = ModelConfig.from_yaml("model_config.yaml")

# Load tokenizer
hartokenizer = RegexTokenizer()
if os.path.exists("models/tokenizer.model"):
    hartokenizer.load("models/tokenizer.model")
else:
    raise AssertionError("Tokenizer needs to be trained")

# Load model
model = GPT(config.vocab_size, config.n_embd, config.n_head, config.n_layer, config.block_size, config.dropout)
model_path = "models/model.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
else:
    raise AssertionError("Model needs to be trained")

# Run inference
prompt = "Hargun Singh Oberoi is"
print(f"Prompt: {prompt}")
prompt_tokens = hartokenizer.encode(prompt)
prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

# Generate text
with torch.no_grad():
    output = model.generate(prompt_tensor, config.max_new_tokens)

# Decode and print output
output_list = output[0].tolist()
output_text = hartokenizer.decode(output_list)
print(f"Completion: {output_text}") 