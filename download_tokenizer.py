"""
This code block allows me to download tokenizer from wandb
"""
import wandb
api = wandb.Api()
artifact = api.artifact("3045/neogpt/tokenizer:latest", type="tokenizer")
artifact_dir = artifact.download(root="models")
