"""
This code block allows me to download data from wandb
"""
#%%
import wandb
api = wandb.Api()
artifact = api.artifact("3045/neogpt/tokenizer:latest", type="tokenizer")
artifact_dir = artifact.download(root="models")
#%%
model_artifact = api.artifact("3045/neogpt/model:latest", type="model")
artifact_dir = model_artifact.download(root="models")