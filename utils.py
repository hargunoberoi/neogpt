"""
Utility classes and functions for model configuration.
"""
from dataclasses import dataclass
from typing import Dict, Any
import yaml
import torch
import os

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
def save_state(model, optimizer, model_dir):
    """
    Save the model and optimizer state.
    """
    save_path = os.path.join(model_dir, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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


def generate_from_model(prompt, model, tokenizer, config, device):
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    output = model.generate(prompt_tensor, config.max_new_tokens)
    output_list = output[0].tolist()
    output_text = tokenizer.decode(output_list)
    return output_text
