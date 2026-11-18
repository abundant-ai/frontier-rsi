"""Unoptimized operation contracts and reproducible workload generation.


"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def attention(q,k,v,allowed=None):
    logits = q.float() @ k.float().transpose(-1,-2) / math.sqrt(q.shape[-1])
    if allowed is not None:
        logits = logits.masked_fill(~allowed, float('-inf'))
    return (logits.softmax(-1) @ v.float()).to(q.dtype)


def reference(kind: str, a: dict[str, torch.Tensor | int | float | bool]) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]:
    """Compute the operation's reference result from the supplied inputs.

    Args:
        kind: Operation identifier for this task.
        a: Input tensors and scalar parameters defined by the task contract.

    Returns:
        Reference tensors in the operation's output order.

    Raises:
        ValueError: The operation identifier does not belong to this task.
    """
    if kind != 'op01':
        raise ValueError(kind)
    if a['q'].shape[-2] >= 4096:
        return F.scaled_dot_product_attention(a['q'], a['k'], a['v'], dropout_p=0.0, is_causal=True)
    return attention(a['q'], a['k'], a['v'], a.get('mask'))


from practice_inputs import inputs


def tensors(result):
    return list(result) if isinstance(result,(list,tuple)) else [result]


def output_and_gradients(fn,a,gradients=False):
    differentiable=[]
    a=dict(a)
    if gradients:
        for key,value in a.items():
            if isinstance(value,torch.Tensor) and value.is_floating_point():
                a[key]=value.detach().clone().requires_grad_(True);differentiable.append(a[key])
    out=tensors(fn(a))
    if any(type(value) is not torch.Tensor for value in out):
        raise ValueError('candidate outputs must be ordinary torch.Tensor values')
    if gradients:
        loss=sum(x.float().square().mean() for x in out)
        grads=torch.autograd.grad(loss,differentiable,allow_unused=False)
        out+=list(grads)
    return [x.detach() for x in out]
