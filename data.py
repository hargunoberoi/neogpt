import os
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
"""
Dataset class for neogpt
"""

class ShardIterableDataset(IterableDataset):
    def __init__(self, data_dir, block_size, split, rank, world_size):
        self.block_size = block_size
        # pick just the shards for this rank
        all_paths = sorted(
            os.path.join(data_dir, fn)
            for fn in os.listdir(data_dir)
            if split in fn
        )
        assert all_paths, f"no shards for {split}"
        # each rank takes every nth file
        self.shard_paths = all_paths[rank :: world_size]
        self.rank = rank

    def __iter__(self):
        for path in self.shard_paths:
            arr = np.load(path, mmap_mode="r")
            n_tokens = arr.shape[0]
            # slide a window of size block_size + 1
            for i in range(0, n_tokens - self.block_size - 1):
                x_np = arr[i : i + self.block_size].copy().astype(np.int32)
                y_np = arr[i + 1 : i + self.block_size + 1].copy().astype(np.int32)
                yield torch.from_numpy(x_np).long(), torch.from_numpy(y_np).long()
            arr._mmap.close()

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
                self._current_arr._mmap.close()
                self._current_arr = None
            path = self.shard_paths[shard_idx]
            self._current_arr = np.load(path, mmap_mode="r")
            self._current_shard_idx = shard_idx

        local_idx = idx - int(self.prefixes[shard_idx])
        arr = self._current_arr
        # force-copy so the tensor doesn’t hold onto the mmap
        x_np = arr[local_idx : local_idx + self.block_size].copy()  
        y_np = arr[local_idx + 1 : local_idx + self.block_size + 1].copy()
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



class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split,master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10b"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt
