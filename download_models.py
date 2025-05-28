"""
This code block allows me to download data from wandb
"""
#%%
import wandb
api = wandb.Api()
download_tokenizer = False
if download_tokenizer:
    artifact = api.artifact("3045/neogpt/tokenizer:latest", type="tokenizer")
    artifact_dir = artifact.download(root="models")
#%%
download_model = True
model_name = "3045/NeoGPT/model_4722794i9amvymi:latest"
if download_model:
    model_artifact = api.artifact(model_name, type="model")
    artifact_dir = model_artifact.download(root="models")
# %%
