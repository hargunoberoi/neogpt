import torch

def estimate_loss(model, iterator, num_iters=10, device='cpu'):
    model.eval()
    losses = torch.zeros(num_iters)
    for k in range(num_iters):
        xb, yb = next(iterator)
        # move to device
        xb,yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        losses[k] = loss.item()
    model.train()
    return losses.mean()
