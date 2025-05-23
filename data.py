import torch
from torch.utils.data import Dataset

"""
Dataset class for neogpt
"""

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        """
        Initialize the dataset with tokens and block size.

        Args:
            tokens (list or tensor): The tokenized data.
            block_size (int): The size of each block (sequence length).
        """
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.tokens) - self.block_size - 1


    def __getitem__(self, idx):
        """
        returns tensors x and y where x are inputs, and ys are targets 
        """
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y
  