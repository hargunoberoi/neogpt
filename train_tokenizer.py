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

#%% # load model_config.yaml and get vocab_size 
import yaml
with open("model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# get vocab_size from config
vocab_size = config["model"]["vocab_size"]

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
# Todo: Write pytest test cases to check if this works
# sample = text[:1000]
# enc = hartokenizer.encode(sample)
# dec = hartokenizer.decode(enc)
# print(f"Encoded: {enc}")
# print(f"Decoded: {dec}")
# %%
