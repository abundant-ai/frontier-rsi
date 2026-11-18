import torch

def configure(shapes):
    return []

def training_loss(forward,inputs,targets,auxiliary):
    logits,hidden=forward(inputs)
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100)
