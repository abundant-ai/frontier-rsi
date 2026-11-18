import torch

def optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=6e-4,weight_decay=.1)
