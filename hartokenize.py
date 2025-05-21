"""
Building the bigram model for practice:
~~But I will use the gpt4 tokenizer (clk1000) because why not?
I had to retrain my own tokenizer because I cannot
have such a large embedding matrix
"""
#%%
# open file input.txt
with open("input.txt", "r") as f:
    text = f.read()
#%%
from harbpe import RegexTokenizer
import os

#%%
V = vocab_size = 512
hartokenizer = RegexTokenizer(max_tokens=vocab_size)
# check if models/tokenizer.model does not exist,
if os.path.exists("models/tokenizer.model"):
    hartokenizer.load("models/tokenizer.model")
else:
    hartokenizer.train(text)
    os.makedirs("models", exist_ok=True)
    prefix = os.path.join("models", "tokenizer")
    hartokenizer.save(prefix)

# %% test tokenizer on small sample

sample = text[:1000]
enc = hartokenizer.encode(sample)
dec = hartokenizer.decode(enc)
print(f"Encoded: {enc}")
print(f"Decoded: {dec}")
# %%
