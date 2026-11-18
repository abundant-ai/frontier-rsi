"""Untimed input families exercising the published operation contracts."""
from __future__ import annotations

from collections.abc import Callable
import torch

Values = dict[str, torch.Tensor | int | float | bool]
Kernel = Callable[[Values], torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]]
LABELS = {
    'op01': ('tail_257', 'tail_8191', 'future_base', 'future_changed', 'signed_gradient', 'sparse_gradient', 'ramp_gradient'),
    'op06': ('allowed_mean', 'blocked_base', 'blocked_changed', 'signed_gradient', 'sparse_gradient', 'ramp_gradient'),
    'op04': ('zero_query_mean', 'permutation_base', 'permutation_changed', 'last_context_token'),
    'op05': ('repeated_page_mean', 'physical_permutation', 'omitted_page_changed', 'last_logical_page'),
    'op13': ('zero_state', 'zero_query', 'projection_tail_256', 'projection_tail_512'),
    'op03': ('last_of_64', 'groups_base', 'groups_permuted'),
    'op11': ('contiguous', 'a_transposed', 'b_transposed', 'both_transposed', 'odd_tail'),
    'op10': ('fp32_decay_100', 'bf16_cross_rounding_100', 'bf16_zero_change_100', 'late_step_100'),
    'op16': ('top_p_below_band', 'top_p_above_band', 'cdf_boundaries', 'k_exceeds_vocabulary'),
}


def random_tensor(shape: tuple[int, ...], seed: int, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Allocate moderate deterministic CUDA values.

    Args:
        shape: Required dimensions.
        seed: Independent tensor seed.
        dtype: Storage type required by the operation.

    Returns:
        A contiguous CUDA tensor with normally distributed values scaled by 0.2.
    """
    generator = torch.Generator(device='cuda').manual_seed(seed)
    return torch.randn(shape, generator=generator, device='cuda', dtype=dtype) * .2


def tensor(values: Values, name: str) -> torch.Tensor:
    """Read one tensor with an explicit type check.

    Args:
        values: Operation inputs.
        name: Tensor field to retrieve.

    Returns:
        The tensor stored at name.

    Raises:
        ValueError: The field is not a tensor.
    """
    value = values[name]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f'{name} must be a tensor')
    return value


def attention_inputs(kind: str, index: int) -> Values:
    """Create attention tails, mask isolation cases and gradient-probe inputs.

    Args:
        kind: Causal or masked attention operation.
        index: Index into LABELS[kind].

    Returns:
        BF16 Q/K/V and a valid causal or block mask.
    """
    lengths = (257, 8191, 513, 513, 257, 383, 1025) if kind == 'op01' else (257, 513, 513, 257, 383, 1025)
    length = lengths[index]
    head_dim = 128 if index == len(lengths)-1 else 64
    shape = (1, 8, length, head_dim)
    values: Values = {name: random_tensor(shape, seed) for name, seed in zip(('q', 'k', 'v'), (9123, 9124, 9125))}
    if kind == 'op01':
        mask = torch.ones(length, length, device='cuda', dtype=torch.bool).tril()
        if index < 2:
            for name in ('q', 'k', 'v'):
                tensor(values, name).zero_()
            tensor(values, 'v')[..., -1, :] = torch.arange(1, 9, device='cuda')[:, None] / 8
        elif index == 3:
            tensor(values, 'k')[..., 256:, :] *= -2
            tensor(values, 'v')[..., 256:, :] += 2
    else:
        blocks = (length + 63)//64
        block_mask = torch.eye(blocks, device='cuda', dtype=torch.bool)
        # A cyclic second block per row stays below 40% when blocks >= 5.
        block_mask[torch.arange(blocks), (torch.arange(blocks)+1)%blocks] = True
        mask = block_mask.repeat_interleave(64, 0).repeat_interleave(64, 1)[:length, :length]
        if index == 0:
            tensor(values, 'q').zero_()
            tensor(values, 'v').copy_((torch.arange(length, device='cuda') % 17)[None, None, :, None] / 16)
        elif index == 2:
            # These positions are blocked for the first query block only.
            tensor(values, 'k')[..., 128:, :] *= -2
            tensor(values, 'v')[..., 128:, :] += 2
    values['mask'] = mask
    return values


def single_query_inputs(index: int) -> Values:
    """Create mean, permutation and isolated final-token cases.

    Args:
        index: Index into the single-query labels.

    Returns:
        Inputs within the declared batch, context and head-dimension ranges.
    """
    batch, length, dim = (3, 1025, 64) if index == 0 else (1, 65535, 64) if index == 3 else (1, 1025, 128)
    values: Values = {'q': random_tensor((batch, 8, 1, dim), 400),
                      'k': random_tensor((batch, 8, length, dim), 401),
                      'v': random_tensor((batch, 8, length, dim), 402)}
    if index == 0:
        tensor(values, 'q').zero_()
        tensor(values, 'v').add_(torch.arange(1, batch+1, device='cuda')[:, None, None, None])
    if index == 2:
        for name in ('k', 'v'):
            values[name] = tensor(values, name).flip(-2).contiguous()
    if index == 3:
        for name in ('q', 'k', 'v'):
            tensor(values, name).zero_()
        tensor(values, 'v')[..., -1, :] = 8
    return values


def fragmented_inputs(index: int) -> Values:
    """Exercise page multiplicities, remapping, omitted data and logical tails.

    Args:
        index: Index into the fragmented-attention labels.

    Returns:
        Whole-page input storage and an int64 logical page table.
    """
    pages = random_tensor((32, 2, 8, 16, 64), 500)
    table = (torch.arange(32, device='cuda') % 7).long()
    query = torch.zeros((1, 8, 1, 64), device='cuda', dtype=torch.bfloat16)
    pages[:, 1] = (torch.arange(32, device='cuda')+1)[:, None, None, None] / 8
    if index == 1:
        pages = pages.flip(0).contiguous()
        table = 31-table
    elif index == 2:
        pages[7:] = -7
    elif index == 3:
        pages.zero_()
        table.zero_()
        table[-1] = 31
        pages[31, 1] = 4
    return {'q': query, 'pages': pages, 'table': table}


def projected_inputs(index: int) -> Values:
    """Separate projection matrices and isolate final latent-coordinate signal.

    Args:
        index: Index into the projected-attention labels.

    Returns:
        Odd-batch, odd-context inputs in the declared latent dimensions.
    """
    latent = 512 if index == 3 else 256
    values: Values = {'q': random_tensor((3, 1, 128), 600),
                      'state': random_tensor((3, 1025, latent), 601),
                      'key_projection': random_tensor((latent, 128), 602),
                      'value_projection': random_tensor((latent, 128), 603)}
    if index == 0:
        tensor(values, 'state').zero_()
    elif index == 1:
        tensor(values, 'q').zero_()
    else:
        for name in ('q', 'state', 'key_projection', 'value_projection'):
            tensor(values, name).zero_()
        tensor(values, 'q')[..., 0] = 2
        tensor(values, 'state')[:, -1, -1] = 4
        tensor(values, 'key_projection')[-1, 0] = 2
        tensor(values, 'value_projection')[-1, 1] = 3
        tensor(values, 'value_projection')[-1, 17] = -1
    return values


def grouped_inputs(index: int) -> Values:
    """Create distinguishable products, including a lone nonzero final group.

    Args:
        index: Index into the independent-matmul labels.

    Returns:
        Sixty-four unequal BF16 products, or their reversed ordering.
    """
    values: Values = {'count': 64}
    for i in range(64):
        group = 63-i if index == 2 else i
        m, k, n = 16+group%5, 17+group%7, 19+group%11
        a = torch.zeros((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.zeros((k, n), device='cuda', dtype=torch.bfloat16)
        if index != 0 or i == 63:
            a[-1, -1] = (group+1)/16
            b[-1, -1] = 2
            if index != 0:
                a[3, 5] = 1
                b[5, 7] = (group+1)/32
        values[f'a{i}'], values[f'b{i}'] = a, b
    return values


def rectangular_inputs(index: int) -> Values:
    """Exercise four stride layouts and odd-size reduction boundaries.

    Args:
        index: Index into the rectangular-matmul labels.

    Returns:
        BF16 matrix operands with interior and boundary signals.
    """
    m, n, k = (65, 97, 129)
    a = torch.zeros((k, m) if index in (1, 3) else (m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.zeros((n, k) if index in (2, 3) else (k, n), device='cuda', dtype=torch.bfloat16)
    if index in (1, 3): a = a.T
    if index in (2, 3): b = b.T
    a[-1, -1], b[-1, -1] = 2, 3
    if index != 4:
        a[3, 7], b[7, 11] = 4, -2
        a[19, 31], b[31, 13] = -1, 2
    return {'a': a, 'b': b}


def optimizer_inputs(index: int) -> Values:
    """Create fixed-gradient trajectories with decay and rounding edge cases.

    Args:
        index: Index into the optimizer labels.

    Returns:
        Eight heterogeneous state groups and their initial positive step.
    """
    dtype = torch.bfloat16 if index in (1, 2) else torch.float32
    values: Values = {'count': 8, 'step': 1000 if index == 3 else 1}
    for i in range(8):
        size = 1024+37*i
        values[f'p{i}'] = torch.full((size,), .03125 if index == 1 else 1., dtype=dtype, device='cuda')
        values[f'g{i}'] = torch.full((size,), .125 if index == 1 else 0., dtype=dtype, device='cuda')
        values[f'm{i}'] = torch.zeros(size, device='cuda', dtype=torch.float32)
        values[f'v{i}'] = torch.zeros(size, device='cuda', dtype=torch.float32)
        if index == 3:
            values[f'g{i}'] = random_tensor((size,), 800+i, dtype=dtype)
            values[f'm{i}'] = random_tensor((size,), 820+i, dtype=torch.float32)
            values[f'v{i}'] = random_tensor((size,), 840+i, dtype=torch.float32).abs()
    return values


def sampling_inputs(index: int) -> Values:
    """Probe cutoff bands, inverse-CDF decisions and unclamped small top-k.

    Args:
        index: Index into the logit-selection labels.

    Returns:
        Logits and uniforms with valid temperature, k and p.
    """
    if index == 3:
        return {'logits': random_tensor((1, 16), 900), 'k': 100, 'p': .9, 'temperature': 1.,
                'uniforms': (torch.arange(1000000, device='cuda', dtype=torch.float64)+.5)[None, :]/1000000}
    logits = torch.full((1, 8192), -8., device='cuda', dtype=torch.bfloat16)
    positions = torch.tensor([3, 111, 709, 1023, 2041, 4097, 6143, 8191], device='cuda')
    logits[:, positions] = 0.
    # Seven of eight equal probabilities have preceding mass 0.875 exactly.
    p = .875 + (-4e-6 if index == 0 else 4e-6) if index < 2 else 1.
    uniforms = torch.tensor([[0., .1, .5, .9, 1.-1e-8]], device='cuda', dtype=torch.float64)
    if index == 2:
        boundaries = torch.arange(1, 8, device='cuda', dtype=torch.float64)/8
        uniforms = (boundaries[:, None] + torch.tensor([-2e-9, 0., 2e-9], device='cuda')).reshape(1, -1)
    return {'logits': logits, 'k': 8, 'p': p, 'temperature': 1., 'uniforms': uniforms}


FACTORIES: dict[str, Callable[[int], Values]] = {
    'op04': single_query_inputs, 'op05': fragmented_inputs, 'op13': projected_inputs,
    'op03': grouped_inputs, 'op11': rectangular_inputs, 'op10': optimizer_inputs, 'op16': sampling_inputs,
}


def inputs(kind: str, index: int) -> Values:
    """Construct one additional case without modifying any timed generator.

    Args:
        kind: Operation identifier.
        index: Index into its fixed untimed case list.

    Returns:
        Fresh input tensors on CUDA.

    Raises:
        ValueError: The case index is outside the registered suite.
    """
    if kind not in LABELS or not 0 <= index < len(LABELS[kind]):
        raise ValueError('Unknown additional correctness case')
    if kind in ('op01', 'op06'):
        return attention_inputs(kind, index)
    return FACTORIES[kind](index)


def evaluate(fn: Kernel, values: Values, label: str) -> list[torch.Tensor]:
    """Evaluate outputs, optionally using a prescribed upstream gradient.

    Args:
        fn: Candidate or independent reference callable.
        values: Input state for a single call.
        label: Registered case name specifying an optional gradient probe.

    Returns:
        Detached outputs followed by Q/K/V gradients when requested.

    Raises:
        ValueError: The result contains non-tensors or the backward signature differs.
    """
    gradients = label.endswith('_gradient')
    values = dict(values)
    if gradients:
        for name in ('q', 'k', 'v'):
            values[name] = tensor(values, name).detach().clone().requires_grad_(True)
    result = fn(values)
    outputs = list(result) if isinstance(result, (tuple, list)) else [result]
    if any(type(value) is not torch.Tensor for value in outputs):
        raise ValueError('Outputs must be ordinary tensors')
    if gradients:
        if len(outputs) != 1: raise ValueError('Attention must return one output')
        output = outputs[0]
        probe = torch.arange(output.numel(), device='cuda').reshape(output.shape)
        if label == 'signed_gradient': probe = (probe % 2)*2-1
        elif label == 'sparse_gradient': probe = ((probe % output.shape[-1]) == 0).to(output.dtype)
        else: probe = (probe % output.shape[-1]+1)/output.shape[-1]
        outputs += list(torch.autograd.grad(output, [tensor(values, name) for name in ('q', 'k', 'v')],
                                           grad_outputs=probe.to(output.dtype), allow_unused=False))
    return [value.detach() for value in outputs]


def advance(values: Values, outputs: list[torch.Tensor]) -> Values:
    """Advance one optimizer trajectory while retaining its fixed gradients.

    Args:
        values: Previous optimizer state.
        outputs: Updated p, m and v tensors for every group.

    Returns:
        The next state with step incremented and independent output storage.
    """
    state = dict(values)
    for i in range(int(values['count'])):
        for offset, key in enumerate(('p', 'm', 'v')):
            state[f'{key}{i}'] = outputs[3*i+offset].clone()
    state['step'] = int(values['step'])+1
    return state
