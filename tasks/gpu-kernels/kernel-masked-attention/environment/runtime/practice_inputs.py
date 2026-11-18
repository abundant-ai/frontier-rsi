"""Generate tensors from public dimension specifications, never stored answers."""
from __future__ import annotations
from pathlib import Path
import torch
from practice_cases import Case, read_cases

Values = dict[str, torch.Tensor | int | float | bool]


def make_inputs(kind: str, case: Case, seed: int, device: str = 'cuda') -> Values:
    """Construct fresh operation inputs using dimensions and a combined seed.

    Args:
        kind: Public operation identifier.
        case: Validated dimensions and scalar settings.
        seed: Sweep seed combined with the case's seed offset.
        device: Allocation device; CPU debugging uses FP32.

    Returns:
        Input dictionary accepted by the public mathematical reference.
    """
    g=torch.Generator(device=device).manual_seed((seed*65537+case['seed'])%2**63)
    dtype=torch.float32 if device=='cpu' else torch.bfloat16

    def rand(*shape: int, fp32: bool = False) -> torch.Tensor:
        """Create moderate normally distributed values.

        Args:
            *shape: Tensor dimensions.
            fp32: Whether to force FP32.

        Returns:
            A newly allocated seeded tensor.
        """
        return torch.randn(shape,generator=g,device=device,dtype=torch.float32 if fp32 else dtype)*.3

    c=case
    if kind=='op03':
        out: Values={'count':len(c['products'])}
        for i,(m,n,k) in enumerate(c['products']): out.update({f'a{i}':rand(m,k),f'b{i}':rand(k,n)})
        return out
    if kind=='op11':
        m,n,k=c['m'],c['n'],c['k']
        return {'a':rand(k,m).T if c['transpose_a'] else rand(m,k),
                'b':rand(n,k).T if c['transpose_b'] else rand(k,n)}
    if kind=='op10':
        out={'count':len(c['sizes']),'step':c['step']}
        for i,size in enumerate(c['sizes']):
            for key in ('p','g','m','v'):
                x=rand(size,fp32=key in ('m','v') or c['dtype']=='fp32')
                out[f'{key}{i}']=x.abs() if key=='v' else x
        return out
    if kind=='op16':
        return {'logits':rand(c['batch'],c['vocab']),
                'uniforms':torch.rand(c['batch'],c['samples'],device=device,generator=g),
                'k':c['k'],'p':c['p'],'temperature':c['temperature']}
    t,d=c['length'],c['head_dim']
    if kind=='op13':
        b,l=c['batch'],c['latent_dim']
        return {'q':rand(b,1,d),'state':rand(b,t,l),'key_projection':rand(l,d),'value_projection':rand(l,d)}
    h=c['heads']
    if kind=='op05':
        page=c['page_size']; n=t//page
        table=torch.randperm(n,device=device,generator=g)
        if c['page_mode']=='repeated': table[0]=table[1]
        return {'q':rand(1,h,1,d),'pages':rand(n,2,h,page,d),'table':table}
    b=c['batch']
    if kind=='op04': return {'q':rand(b,h,1,d),'k':rand(b,h,t,d),'v':rand(b,h,t,d)}
    block=c['block_size']; n=(t+block-1)//block
    diagonal=torch.eye(n,device=device,dtype=torch.bool)
    active=max(n,min(n*n,round(c['density']*n*n)))
    mask=diagonal.flatten().clone()
    candidates=torch.arange(n*n,device=device)[~mask]
    order=torch.randperm(candidates.numel(),device=device,generator=g)
    mask[candidates[order[:active-n]]]=True
    return {'q':rand(b,h,t,d),'k':rand(b,h,t,d),'v':rand(b,h,t,d),
            'mask':mask.reshape(n,n).repeat_interleave(block,0).repeat_interleave(block,1)[:t,:t].contiguous()}


def inputs(kind: str, seed: int, index: int, device: str = 'cpu', smoke: bool = False) -> Values:
    """Generate one built-in example through the operation runtime interface.

    Args:
        kind: Operation identifier.
        seed: Controls values without changing dimensions.
        index: Position in the public example list.
        device: Torch allocation device.
        smoke: Uses the smallest public example when requested.

    Returns:
        Fresh inputs for the requested example.
    """
    cases=read_cases(Path('/app/practice_cases.json'),kind)
    return make_inputs(kind,cases[0 if smoke else index],seed,device)
