import math
import sys
import time
import torch


def _duration():
    # Default to the final 30-minute budget; development passes --seconds.
    try:
        return float(sys.argv[sys.argv.index('--seconds') + 1])
    except (ValueError, IndexError):
        return 1800.0


@torch.no_grad()
def _orthogonalize(g, steps=5):
    # Approximate the polar factor using the smaller Gram matrix.
    x = g.to(torch.bfloat16)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        a = x @ x.transpose(-2, -1)
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    return x.transpose(-2, -1) if transposed else x


class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, groups):
        super().__init__(groups, dict(lr=0.002))
        self.adam = torch.optim.AdamW(
            [g for g in self.param_groups if g['kind'] == 'adam'],
            betas=(0.9, 0.995), eps=1e-8, fused=True)
        self.started = time.monotonic()
        self.duration = max(60.0, _duration() - 20.0)
        self.updates = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        progress = min(1.0, (time.monotonic() - self.started) / self.duration)
        scale = (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))
        scale *= min(1.0, (self.updates + 1) / 5.0)
        for group in self.param_groups:
            lr = group['base_lr'] * scale
            group['lr'] = lr
            group['weight_decay'] = group['base_decay']
            if group['kind'] == 'adam':
                if self.updates == 0 and group.get('embedding', False):
                    # Decay the unit-scale embeddings after the first gradient.
                    group['weight_decay'] = 0.98 / lr
                continue
            params = group['params']
            gradients = torch.stack([p.grad for p in params])
            state = self.state[params[0]]
            if 'momentum' not in state:
                state['momentum'] = torch.zeros_like(gradients)
            momentum = state['momentum']
            momentum.lerp_(gradients, 0.10)
            gradients.lerp_(momentum, 0.90)
            if group.get('split_qkv', False):
                # Precondition Q, K, and V independently.
                width = params[0].shape[1]
                update = _orthogonalize(gradients.reshape(-1, width, width)).reshape_as(gradients)
            else:
                update = _orthogonalize(gradients)
                lr *= math.sqrt(max(1.0, params[0].shape[0] / params[0].shape[1]))
            for p, u in zip(params, update.unbind(0)):
                p.mul_(1.0 - group['lr'] * group['weight_decay'])
                p.add_(u, alpha=-lr)
        self.adam.step()
        self.updates += 1
        return loss


def optimizer(model):
    # Batch layer matrices by shape; reserve AdamW for embeddings and vectors.
    matrices = {}
    adam = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        embedding = name in ('wte.weight', 'wpe.weight')
        if p.ndim == 2 and not embedding:
            matrices.setdefault((tuple(p.shape), name.endswith('qkv.weight')), []).append(p)
        else:
            lr = 0.003 if embedding else 0.002
            decay = 0.1 if embedding else 0.0
            key = (lr, decay, embedding)
            if key not in adam:
                adam[key] = dict(params=[], kind='adam', lr=lr, base_lr=lr,
                                 weight_decay=decay, base_decay=decay,
                                 embedding=embedding)
            adam[key]['params'].append(p)
    groups = [dict(params=ps, kind='muon', lr=0.02, base_lr=0.02,
                   weight_decay=0.1, base_decay=0.1, split_qkv=split)
              for (shape, split), ps in matrices.items()]
    return HybridOptimizer(groups + list(adam.values()))


