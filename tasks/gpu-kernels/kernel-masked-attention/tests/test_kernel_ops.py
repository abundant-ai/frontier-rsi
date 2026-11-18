"""Operation contracts and reproducible workload generation."""
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
    if kind != 'op06':
        raise ValueError(kind)
    return attention(a['q'], a['k'], a['v'], a['mask'])


def inputs(kind: str, seed: int, index: int, device: str='cpu', smoke: bool=False) -> dict[str, torch.Tensor | int | float | bool]:
    """Construct a reproducible input case for this operation.

    Args:
        kind: Operation identifier for this task.
        seed: Random seed used to populate the input tensors.
        index: Case index selecting dimensions and scalar parameters.
        device: Device on which to allocate tensors.
        smoke: Use the smaller debugging inputs.

    Returns:
        Input tensors and scalar parameters accepted by the reference function.

    Raises:
        ValueError: The operation identifier does not belong to this task.
    """
    if kind != 'op06':
        raise ValueError(kind)
    g = torch.Generator(device=device).manual_seed(seed * 1009 + index)
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16

    def rand(*shape: int) -> torch.Tensor:
        """Allocate seeded normal values in the case's dtype and device.

        Args:
            *shape: Dimensions of the requested tensor.

        Returns:
            A tensor of normally distributed values scaled by 0.3.
        """
        return torch.randn(shape, generator=g, device=device, dtype=dtype) * 0.3
    t = (16, 24, 32)[index % 3] if smoke else (256, 512, 1024, 2048, 4096, 8192)[index % 6]
    h = 2 if smoke else (8, 16)[index % 2]
    d = 8 if smoke else (64, 128)[index % 2]
    b = 1 if smoke else 1 + index % 4
    qt = t
    if not smoke and False:
        t = (1024, 2048, 4096, 8192, 16384, 32768, 65536)[index % 7]
        b = (1, 2, 4, 8, 16, 32)[index % 6]
        while b > 1 and b * h * t * d * 4 > 4 * 1024 ** 3:
            b //= 2
    if not smoke:
        while b > 1 and b * h * t * t * 16 > 16 * 1024 ** 3:
            b //= 2
    out = {'q': rand(b, h, qt, d), 'k': rand(b, h, t, d), 'v': rand(b, h, t, d)}
    block = 4 if smoke else (64, 128)[index // 2 % 2]
    n = (t + block - 1) // block
    target_density = (0.1, 0.2, 0.3, 0.4)[index % 4]
    active = max(n, min(math.floor(0.4 * n * n), round(target_density * n * n)))
    mask = torch.eye(n, device=device, dtype=torch.bool).reshape(-1)
    off_diagonal = torch.arange(n * n, device=device).reshape(n, n)
    off_diagonal = off_diagonal[~torch.eye(n, device=device, dtype=torch.bool)]
    chosen = off_diagonal[torch.randperm(off_diagonal.numel(), generator=g, device=device)[:active - n]]
    mask[chosen] = True
    mask = mask.reshape(n, n)
    out['mask'] = mask.repeat_interleave(block, 0).repeat_interleave(block, 1)[:t, :t]
    return out


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
