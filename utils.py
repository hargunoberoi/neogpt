"""
Utility classes and functions for model configuration.
"""
from dataclasses import dataclass
from typing import Dict, Any
import yaml
import torch
import numpy as np
import os
import math
@dataclass
class ModelConfig:
    """
    Configuration class for model hyperparameters.
    Takes a dictionary and sets attributes using key-value pairs.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize ModelConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        # Flatten nested dictionaries and set attributes
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # For nested dictionaries, set each sub-key as an attribute
                for sub_key, sub_value in value.items():
                    setattr(self, sub_key, sub_value)
            else:
                setattr(self, key, sub_value)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """
        Create ModelConfig from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            ModelConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"ModelConfig({', '.join(attrs)})" 

# save model state
def save_state(iteration, model, optimizer, model_dir):
    """
    Save the model and optimizer state.
    """
    save_path = os.path.join(model_dir, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, save_path)
    print(f"Model saved to {save_path}")

def load_state(model, optimizer, model_dir='models'):
    """
    Load the model and optimizer state.
    """
    # get the latest model
    model_file = model_dir + "/model.pth"
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {model_file}")


def estimate_loss(model, iterator, num_iters=10, device='cpu'):
    model.eval()
    losses = torch.zeros(num_iters)
    for k in range(num_iters):
        xb, yb = next(iterator)
        # move to device
        xb,yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def generate_from_model(prompt, model, tokenizer,device, max_new_tokens=100):
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    output = model.generate(prompt_tensor, max_new_tokens)
    output_list = output[0].tolist()
    output_text = tokenizer.decode(output_list)
    return output_text

def get_lr(iteration, warmup_iters, max_lr, min_lr, max_iters):
    #1) linear warmup for warmup_iters
    if iteration < warmup_iters:
        return max_lr * (iteration + 1) / (warmup_iters + 1)
    
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > max_iters:
        return min_lr
    
    #3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1, "Decay ratio must be between 0 and 1"
    coeff = 0.5 * (1. + math.cos(math.pi*decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def load_tokens(filename):
    np_tokens = np.load(filename)
    return np_tokens