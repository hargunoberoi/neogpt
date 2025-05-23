"""
Utility classes and functions for model configuration.
"""
from dataclasses import dataclass
from typing import Dict, Any
import yaml


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