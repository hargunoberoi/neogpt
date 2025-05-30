import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
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
  
class StreamingTextDataset(IterableDataset):
    def __init__(self,block_size,tokenizer, rank=0,world_size=1):
        self.ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        self.tokenizer = tokenizer
        # later figure out how to put a endoftexttokenizer in my tokenizer
        self.block_size = block_size
        self.buffer = []
        self.rank = rank
        self.world_size = world_size

    def __iter__(self): 
        for i,sample in enumerate(self.ds):
            if i % self.world_size != self.rank:
                continue
            
            text = sample["text"]
            tokens = self.tokenizer.encode(text)
            self.buffer.extend(tokens)

            while len(self.buffer) >= self.block_size + 1:
                x = self.buffer[:self.block_size]
                y = self.buffer[1:self.block_size + 1]
                yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
                self.buffer = self.buffer[1:]