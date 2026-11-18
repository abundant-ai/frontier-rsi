"""Validate public custom cases against each operation's documented input domain."""
from __future__ import annotations
import json
import math
from pathlib import Path

Case = dict[str, object]


def integer(value: object, low: int, high: int, name: str) -> int:
    """Require a bounded integer, excluding JSON booleans.

    Args:
        value: Decoded field.
        low: Inclusive minimum.
        high: Inclusive maximum.
        name: Field label for errors.

    Returns:
        The validated integer.

    Raises:
        ValueError: The type or range is invalid.
    """
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}')
    return value


def validate_cases(raw: object, kind: str) -> list[Case]:
    """Check shapes and scalar settings before allocating any tensors.

    Args:
        raw: JSON list of one to 100 shape dictionaries.
        kind: Operation identifier from the public contract.

    Returns:
        Validated specifications, including default seed offsets.

    Raises:
        ValueError: A required field, range or relation is invalid.
    """
    fields = {
        'op03': {'products'}, 'op04': {'batch','heads','length','head_dim'},
        'op05': {'heads','length','head_dim','page_size','page_mode'},
        'op06': {'batch','heads','length','head_dim','block_size','density'},
        'op10': {'sizes','dtype','step'}, 'op13': {'batch','length','latent_dim','head_dim'},
        'op16': {'batch','vocab','k','p','temperature','samples'},
        'op11': {'m','n','k','transpose_a','transpose_b'},
    }
    if kind not in fields or not isinstance(raw,list) or not 1 <= len(raw) <= 100:
        raise ValueError('Expected 1 to 100 cases for a supported operation')
    result: list[Case] = []
    for index,item in enumerate(raw):
        required=fields[kind]
        if not isinstance(item,dict) or not required <= item.keys() or item.keys()-required-{'seed'}:
            raise ValueError(f'Case {index} requires {sorted(required)} and optional seed')
        c=dict(item,seed=integer(item.get('seed',index),0,2**63-1,'seed'))
        if kind in ('op04','op05','op06','op13'):
            bounds={'op04':(1024,65536),'op05':(512,32768),'op06':(256,8192),'op13':(1024,32768)}
            integer(c['length'],*bounds[kind],'length')
            # Head counts and widths without a stated upper bound remain positive.
            if 'heads' in c: integer(c['heads'],1,2**31-1,'heads')
            integer(c['head_dim'],1,2**31-1,'head_dim')
            if kind == 'op04' and c['head_dim'] not in (64,128):
                raise ValueError('head_dim must be 64 or 128')
            if 'batch' in c: integer(c['batch'],1,32 if kind in ('op04','op13') else 2**31-1,'batch')
        if kind=='op05':
            if c['page_size'] not in (16,32) or type(c['page_size']) is not int or c['length']%c['page_size']:
                raise ValueError('length must contain whole pages of 16 or 32 tokens')
            if c['page_mode'] not in ('permutation','repeated'): raise ValueError('Unknown page_mode')
        if kind=='op06':
            if type(c['block_size']) is not int or c['block_size'] not in (64,128): raise ValueError('Invalid block_size')
            if type(c['density']) not in (int,float) or not .1<=c['density']<=.4: raise ValueError('density must be 0.1 to 0.4')
        if kind=='op13':
            if type(c['latent_dim']) is not int or c['latent_dim'] not in (256,512): raise ValueError('latent_dim must be 256 or 512')
        if kind=='op03':
            products=c['products']
            if not isinstance(products,list) or not 2<=len(products)<=64: raise ValueError('Provide 2 to 64 products')
            for shape in products:
                if not isinstance(shape,list) or len(shape)!=3: raise ValueError('Products use [M,N,K]')
                for n in shape: integer(n,16,2048,'matrix dimension')
        if kind=='op11':
            for f in ('m','n','k'): integer(c[f],64,8192,f)
            if any(type(c[f]) is not bool for f in ('transpose_a','transpose_b')): raise ValueError('Transpose flags must be Boolean')
        if kind=='op10':
            if not isinstance(c['sizes'],list) or not 8<=len(c['sizes'])<=15: raise ValueError('Provide 8 to 15 tensor sizes')
            for size in c['sizes']: integer(size,1024,8000000,'tensor size')
            if c['dtype'] not in ('bf16','fp32'): raise ValueError('dtype must be bf16 or fp32')
            integer(c['step'],1,2**31-1,'step')
        if kind=='op16':
            integer(c['batch'],1,256,'batch'); v=integer(c['vocab'],16,131072,'vocab')
            if v!=16 and v<8192: raise ValueError('vocab must be 16 or 8192 to 131072')
            integer(c['k'],1,100,'k'); integer(c['samples'],1,2**31-1,'samples')
            if type(c['p']) not in (int,float) or not .8<=c['p']<=1: raise ValueError('p must be 0.8 to 1')
            if type(c['temperature']) not in (int,float) or not math.isfinite(c['temperature']) or c['temperature']<=0: raise ValueError('temperature must be positive and finite')
            if v==16 and c['samples']!=1000000: raise ValueError('16-token cases require one million samples')
        result.append(c)
    return result


def read_cases(path: Path, kind: str) -> list[Case]:
    """Load a bounded JSON case list.

    Args:
        path: File of public specifications.
        kind: Operation from the public contract.

    Returns:
        Validated cases.

    Raises:
        ValueError: The file exceeds 1 MiB or fails schema validation.
    """
    if path.stat().st_size>1024**2: raise ValueError('Case file exceeds 1 MiB')
    return validate_cases(json.loads(path.read_text()),kind)
