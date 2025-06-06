"""
This code block allows me to download data from wandb
"""
#%%
import wandb
api = wandb.Api()
model_name = "3045/NeoGPT/fineweb-edu:latest"
model_artifact = api.artifact(model_name, type="dataset")
artifact_dir = model_artifact.download(root=".")
# %%
