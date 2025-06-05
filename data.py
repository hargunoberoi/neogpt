import os
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
"""
Dataset class for neogpt
"""

class ShardDataset(Dataset):
    def __init__(self, data_dir, block_size, split="train"):
        assert split in ["train","val"]
        self.block_size = block_size
        # find all shard filenames (don’t load them yet)
        shards = sorted(
            os.path.join(data_dir, fn)
            for fn in os.listdir(data_dir)
            if split in fn
        )
        assert shards, f"no shards for {split}"
        self.shard_paths = shards
        # compute counts and prefixes once
        counts = []
        for path in self.shard_paths:
            arr = np.load(path, mmap_mode="r")
            counts.append(arr.shape[0] - block_size - 1)
            del arr
        self.counts = np.array(counts, dtype=np.int64)
        self.prefixes = np.concatenate(([0], np.cumsum(self.counts[:-1])))
        # only one memmap open at a time:
        self._current_shard_idx = None
        self._current_arr = None

    def __len__(self):
        return int(self.prefixes[-1] + self.counts[-1])

    def __getitem__(self, idx):
        shard_idx = int(np.searchsorted(self.prefixes, idx, side="right") - 1)
        if shard_idx != self._current_shard_idx:
            if self._current_arr is not None:
                # close the old mmap explicitly
                self._current_arr._mmap.close()
                self._current_arr = None
            path = self.shard_paths[shard_idx]
            arr = np.load(path, mmap_mode="r")
            self._current_arr = arr
            self._current_shard_idx = shard_idx

        local_idx = idx - int(self.prefixes[shard_idx])
        arr = self._current_arr
        x_np = arr[local_idx : local_idx + self.block_size]
        y_np = arr[local_idx + 1 : local_idx + self.block_size + 1]
        return torch.from_numpy(x_np).long(), torch.from_numpy(y_np).long()


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