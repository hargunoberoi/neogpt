"""
This code block allows me to download data from wandb
"""
#%%
import wandb
api = wandb.Api()
download_tokenizer = True
if download_tokenizer:
    artifact = api.artifact("3045/neogpt/tokenizer:latest", type="tokenizer")
    artifact_dir = artifact.download(root="models")
#%%
download_model = False
if download_model:
    model_artifact = api.artifact("3045/neogpt/model:latest", type="model")
    artifact_dir = model_artifact.download(root="models")
# %%
