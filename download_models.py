"""
This code block allows me to download data from wandb
"""
#%%
import wandb
api = wandb.Api()
model_name = "3045/NeoGPT/model_47227942unhl444:latest"
model_artifact = api.artifact(model_name, type="model")
artifact_dir = model_artifact.download(root="models")
# %%
