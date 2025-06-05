import os
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
"""
Dataset class for neogpt
"""

class ShardDataset(Dataset):

    def __init__(self,data_dir,block_size, split="train"):
        assert split in ["train", "val"], "split must be either 'train' or 'val'"
        self.block_size = block_size
        self.shards = self._get_filenames(data_dir, split)
        self.mmaps = [np.load(path, mmap_mode='r') for path in self.shards]
        self.counts = np.array([mm.shape[0] - self.block_size - 1 for mm in self.mmaps], dtype=np.int64)
        self.prefixes = np.concatenate(([0], np.cumsum(self.counts[:-1])))

    def _get_filenames(self,data_root,split):
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"No shards found for split {split} in {data_root}"
        return shards
    
    def __len__(self):
        return int(self.prefixes[-1] + self.counts[-1])

    def __getitem__(self, idx):
        # find the shard that contains the idx
        shard_idx = np.searchsorted(self.prefixes, idx, side='right') - 1
        assert shard_idx >= 0, "Index out of bounds"
        assert shard_idx < len(self.shards), "Shard index out of bounds"
        
        local_idx = idx - int(self.prefixes[shard_idx])

        arr = self.mmaps[shard_idx]
        # get the input and target tokens
        x_np = arr[local_idx:local_idx + self.block_size]
        y_np = arr[local_idx + 1:local_idx + self.block_size + 1]

        return torch.from_numpy(x_np).long(), torch.from_numpy(y_np).long()
    
    def _get_num_tokens(self):
        counts = np.zeros(len(self.shards), dtype=np.int64)
        prefixes = np.zeros(len(self.shards), dtype=np.int64)
        for idx,path in enumerate(self.shards):
            np_tokens = np.load(path,mmap_mode='r')
            shard_count = np_tokens.shape[0] - self.block_size - 1
            counts[idx] = shard_count
            prefixes[idx] = np.sum(counts[:idx])
        return counts, prefixes

    def load_tokens(self,filename):
        np_tokens = np.load(filename)
        torch_tokens = torch.tensor(np_tokens, dtype=torch.long)
        return np_tokens

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