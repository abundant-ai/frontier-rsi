"""Centered online-softmax attention with FP32 probability and gradient products."""
import torch
import triton
import triton.language as tl

PRECISION_TERMS = 3
CAUSAL = False
FORWARD_M = 64
FORWARD_N = 64
FORWARD_WARPS = 4
BACKWARD_TILE = 64
BACKWARD_WARPS = 4


@triton.jit
def _split_dot(A: tl.tensor, B: tl.tensor, LEFT: tl.constexpr, RIGHT: tl.constexpr) -> tl.tensor:
    """Multiply FP32 operands through explicitly retained BF16 residual parts.

    Args:
        A, B: Compatible FP32 matrix tiles.
        LEFT: Number of BF16 parts retained from A, from one to three.
        RIGHT: Number of BF16 parts retained from B, one or two.

    Returns:
        FP32 sum of all retained cross-products on BF16 tensor cores.
    """
    a0 = A.to(tl.bfloat16)
    b0 = B.to(tl.bfloat16)
    result = tl.dot(a0, b0)
    if LEFT >= 2:
        a1 = (A - a0.to(tl.float32)).to(tl.bfloat16)
        result += tl.dot(a1, b0)
    if LEFT >= 3:
        a2 = (A - a0.to(tl.float32) - a1.to(tl.float32)).to(tl.bfloat16)
        result += tl.dot(a2, b0)
    if RIGHT >= 2:
        b1 = (B - b0.to(tl.float32)).to(tl.bfloat16)
        result += tl.dot(a0, b1)
        if LEFT >= 2:
            result += tl.dot(a1, b1)
        if LEFT >= 3:
            result += tl.dot(a2, b1)
    return result


@triton.jit
def _forward(Q: tl.tensor, K: tl.tensor, V: tl.tensor, Mask: tl.tensor,
             O: tl.tensor, OF: tl.tensor, L: tl.tensor,
             N: int, D: tl.constexpr, DB: tl.constexpr, H: tl.constexpr,
             SMB: tl.constexpr, SMH: tl.constexpr, SM: int, SN: tl.constexpr,
             CAUSE: tl.constexpr, PT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr) -> None:
    """Compute an output tile and its log-normalizer without score materialization.

    Args:
        Q, K, V: Contiguous BF16 input pointers.
        Mask: Shared or broadcastable dense Boolean block mask.
        O: BF16 output pointer.
        OF: FP32 online mean of half-scaled values centered on the first vector.
        L: FP32 base-two log-normalizers.
        N, D, DB, H: Sequence length, actual/padded head width, and head count.
        SMB, SMH, SM, SN: Broadcast-aware mask strides.
        PT: Number of BF16 residual parts retained for probabilities and score derivatives.
        CAUSE: Whether the public mask is lower triangular.
        BM, BN: Query and key tile sizes.
    """
    row0 = tl.program_id(0) * BM
    head = tl.program_id(1)
    rows = row0 + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    ds = tl.arange(0, DB)
    base = head * N * D
    maskbase = (head // H) * SMB + (head % H) * SMH
    anchor = tl.load(V + base + ds, ds < D, 0).to(tl.float32) * 0.5
    q = tl.load(Q + base + rows[:, None] * D + ds[None, :], (rows[:, None] < N) & (ds[None, :] < D), 0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DB), tl.float32)
    stop = N
    if CAUSE:
        stop = tl.minimum(N, row0 + BM)
    for start in range(0, stop, BN):
        active = True
        if not CAUSE:
            active = tl.load(Mask + maskbase + row0 * SM + start * SN)
        if active:
            keys = start + cols
            k = tl.load(K + base + keys[None, :] * D + ds[:, None], (keys[None, :] < N) & (ds[:, None] < D), 0)
            scores = tl.dot(q, k) * (1.4426950408889634 / (D ** 0.5))
            valid = keys[None, :] < N
            if CAUSE:
                valid = valid & (keys[None, :] <= rows[:, None])
            scores = tl.where(valid, scores, -float('inf'))
            updated = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.exp2(maximum - updated)
            probability = tl.exp2(scores - updated[:, None])
            previous_mass = total * alpha
            updated_total = previous_mass + tl.sum(probability, 1)
            acc = acc * (previous_mass / updated_total)[:, None]
            probability = probability / updated_total[:, None]
            total = updated_total
            v = tl.load(V + base + keys[:, None] * D + ds[None, :], (keys[:, None] < N) & (ds[None, :] < D), 0)
            centered = v.to(tl.float32) * 0.5 - anchor[None, :]
            acc += _split_dot(probability, centered, PT, 2)
            maximum = updated
    result = acc
    tl.store(O + base + rows[:, None] * D + ds[None, :], (result + anchor[None, :]) * 2.0, (rows[:, None] < N) & (ds[None, :] < D))
    tl.store(OF + base + rows[:, None] * D + ds[None, :], result, (rows[:, None] < N) & (ds[None, :] < D))
    tl.store(L + head * N + rows, maximum + tl.log2(total), rows < N)


@triton.jit
def _delta(OF: tl.tensor, DO: tl.tensor, Delta: tl.tensor,
           N: int, D: tl.constexpr, DB: tl.constexpr, BM: tl.constexpr) -> None:
    """Reduce the output-gradient product for each softmax row.

    Args:
        OF: Online mean of half-scaled values centered on the first vector.
        DO: Upstream BF16 derivative.
        Delta: FP32 per-row result.
        N, D, DB, BM: Sequence length, actual/padded feature width, and row tile size.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    ds = tl.arange(0, DB)
    offsets = head * N * D + rows[:, None] * D + ds[None, :]
    out = tl.load(OF + offsets, (rows[:, None] < N) & (ds[None, :] < D), 0)
    grad = tl.load(DO + offsets, (rows[:, None] < N) & (ds[None, :] < D), 0).to(tl.float32)
    tl.store(Delta + head * N + rows, tl.sum(out * grad, 1), rows < N)


@triton.jit
def _backward_q(Q: tl.tensor, K: tl.tensor, V: tl.tensor, Mask: tl.tensor,
                DO: tl.tensor, L: tl.tensor, Delta: tl.tensor, DQ: tl.tensor,
                N: int, D: tl.constexpr, DB: tl.constexpr, H: tl.constexpr,
                SMB: tl.constexpr, SMH: tl.constexpr, SM: int, SN: tl.constexpr,
                CAUSE: tl.constexpr, PT: tl.constexpr, BLOCK: tl.constexpr) -> None:
    """Accumulate query gradients with one owner per query tile.

    Args:
        Q, K, V, Mask: Inputs preserved by autograd.
        DO: Upstream derivative.
        L, Delta: Saved normalizers and row derivative sums.
        DQ: Query-gradient output.
        N, D, DB, H: Tensor dimensions, including the padded feature width.
        SMB, SMH, SM, SN: Broadcast-aware mask strides.
        PT: Number of BF16 residual parts retained for probabilities and score derivatives.
        CAUSE: Whether to traverse only causal keys.
        BLOCK: Square score-tile size.
    """
    row0 = tl.program_id(0) * BLOCK
    head = tl.program_id(1)
    rows = row0 + tl.arange(0, BLOCK)
    ds = tl.arange(0, DB)
    base = head * N * D
    maskbase = (head // H) * SMB + (head % H) * SMH
    q = tl.load(Q + base + rows[:, None] * D + ds[None, :], (rows[:, None] < N) & (ds[None, :] < D), 0)
    do = tl.load(DO + base + rows[:, None] * D + ds[None, :], (rows[:, None] < N) & (ds[None, :] < D), 0)
    lse = tl.load(L + head * N + rows, rows < N, 0)
    delta = tl.load(Delta + head * N + rows, rows < N, 0)
    anchor = tl.load(V + base + ds, ds < D, 0).to(tl.float32) * 0.5
    dq = tl.zeros((BLOCK, DB), tl.float32)
    stop = N
    if CAUSE:
        stop = tl.minimum(N, row0 + BLOCK)
    for start in range(0, stop, BLOCK):
        active = True
        if not CAUSE:
            active = tl.load(Mask + maskbase + row0 * SM + start * SN)
        if active:
            keys = start + tl.arange(0, BLOCK)
            k = tl.load(K + base + keys[:, None] * D + ds[None, :], (keys[:, None] < N) & (ds[None, :] < D), 0)
            v = tl.load(V + base + keys[:, None] * D + ds[None, :], (keys[:, None] < N) & (ds[None, :] < D), 0)
            scores = tl.dot(q, tl.trans(k)) * (1.4426950408889634 / (D ** 0.5))
            valid = keys[None, :] < N
            if CAUSE:
                valid = valid & (keys[None, :] <= rows[:, None])
            p = tl.exp2(tl.where(valid, scores - lse[:, None], -float('inf')))
            centered = v.to(tl.float32) * 0.5 - anchor[None, :]
            dp = _split_dot(do.to(tl.float32), tl.trans(centered), 1, 2)
            dscores = (2.0 * p) * (dp - delta[:, None])
            dq += _split_dot(dscores, k.to(tl.float32), PT, 1) * (1.0 / (D ** 0.5))
    tl.store(DQ + base + rows[:, None] * D + ds[None, :], dq, (rows[:, None] < N) & (ds[None, :] < D))


@triton.jit
def _backward_kv(Q: tl.tensor, K: tl.tensor, V: tl.tensor, Mask: tl.tensor,
                 DO: tl.tensor, L: tl.tensor, Delta: tl.tensor,
                 DK: tl.tensor, DV: tl.tensor,
                 N: int, D: tl.constexpr, DB: tl.constexpr, H: tl.constexpr,
                 SMB: tl.constexpr, SMH: tl.constexpr, SM: int, SN: tl.constexpr,
                 CAUSE: tl.constexpr, PT: tl.constexpr, BLOCK: tl.constexpr) -> None:
    """Accumulate key/value gradients without atomics or full score storage.

    Args:
        Q, K, V, Mask: Inputs preserved by autograd.
        DO: Upstream derivative.
        L, Delta: Per-query normalizers and derivative sums.
        DK, DV: Key and value gradient outputs.
        N, D, DB, H: Tensor dimensions, including the padded feature width.
        SMB, SMH, SM, SN: Broadcast-aware mask strides.
        PT: Number of BF16 residual parts retained for probabilities and score derivatives.
        CAUSE: Whether earlier query blocks can be skipped.
        BLOCK: Square score-tile size.
    """
    key0 = tl.program_id(0) * BLOCK
    head = tl.program_id(1)
    keys = key0 + tl.arange(0, BLOCK)
    ds = tl.arange(0, DB)
    base = head * N * D
    maskbase = (head // H) * SMB + (head % H) * SMH
    k = tl.load(K + base + keys[:, None] * D + ds[None, :], (keys[:, None] < N) & (ds[None, :] < D), 0)
    v = tl.load(V + base + keys[:, None] * D + ds[None, :], (keys[:, None] < N) & (ds[None, :] < D), 0)
    anchor = tl.load(V + base + ds, ds < D, 0).to(tl.float32) * 0.5
    dk = tl.zeros((BLOCK, DB), tl.float32)
    dv = tl.zeros((BLOCK, DB), tl.float32)
    begin = 0
    if CAUSE:
        begin = key0
    for start in range(begin, N, BLOCK):
        active = True
        if not CAUSE:
            active = tl.load(Mask + maskbase + start * SM + key0 * SN)
        if active:
            rows = start + tl.arange(0, BLOCK)
            q = tl.load(Q + base + rows[:, None] * D + ds[None, :], (rows[:, None] < N) & (ds[None, :] < D), 0)
            do = tl.load(DO + base + rows[:, None] * D + ds[None, :], (rows[:, None] < N) & (ds[None, :] < D), 0)
            lse = tl.load(L + head * N + rows, rows < N, 0)
            delta = tl.load(Delta + head * N + rows, rows < N, 0)
            scores = tl.dot(q, tl.trans(k)) * (1.4426950408889634 / (D ** 0.5))
            valid = rows[:, None] < N
            if CAUSE:
                valid = valid & (keys[None, :] <= rows[:, None])
            p = tl.exp2(tl.where(valid, scores - lse[:, None], -float('inf')))
            dv += _split_dot(tl.trans(p), do.to(tl.float32), PT, 1)
            centered = v.to(tl.float32) * 0.5 - anchor[None, :]
            dp = _split_dot(do.to(tl.float32), tl.trans(centered), 1, 2)
            dscores = (2.0 * p) * (dp - delta[:, None])
            dk += _split_dot(tl.trans(dscores), q.to(tl.float32), PT, 1) * (1.0 / (D ** 0.5))
    tl.store(DK + base + keys[:, None] * D + ds[None, :], dk, (keys[:, None] < N) & (ds[None, :] < D))
    tl.store(DV + base + keys[:, None] * D + ds[None, :], dv, (keys[:, None] < N) & (ds[None, :] < D))


def _mask_strides(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """Return dense-mask strides with zero strides for broadcast dimensions.

    Args:
        mask: Two-, three-, or four-dimensional attention mask.

    Returns:
        Batch, head, query, and key strides for broadcast indexing.
    """
    batch_stride = 0
    head_stride = 0
    if mask.ndim == 4 and mask.shape[0] != 1:
        batch_stride = mask.stride(0)
    if mask.ndim >= 3 and mask.shape[-3] != 1:
        head_stride = mask.stride(-3)
    return batch_stride, head_stride, mask.stride(-2), mask.stride(-1)


class _Attention(torch.autograd.Function):
    """Preserve only linear-size state for first-order attention gradients."""

    @staticmethod
    def forward(ctx: torch.autograd.Function, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run tiled attention and retain tensors needed by the backward pass.

        Args:
            ctx: Autograd context owning the saved tensors.
            q, k, v: BF16 tensors with shape batch, heads, sequence, feature.
            mask: Boolean causal or block mask.

        Returns:
            BF16 attention output with Q's shape.
        """
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        b, h, n, d = q.shape
        out = torch.empty_like(q)
        full = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        lse = torch.empty((b * h, n), dtype=torch.float32, device=q.device)
        smb, smh, sm, sn = _mask_strides(mask)
        _forward[(triton.cdiv(n, FORWARD_M), b * h)](
            q, k, v, mask, out, full, lse, n, d, max(16, triton.next_power_of_2(d)), h, smb, smh, sm, sn,
            CAUSAL, PRECISION_TERMS, FORWARD_M, FORWARD_N, num_warps=FORWARD_WARPS, num_stages=2)
        ctx.save_for_backward(q, k, v, mask, full, lse)
        return out

    @staticmethod
    def backward(ctx: torch.autograd.Function, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Recompute probabilities tilewise and return Q, K, and V derivatives.

        Args:
            ctx: Context populated by the forward pass.
            grad: Upstream derivative matching the output shape.

        Returns:
            Q, K, V gradients and no gradient for the Boolean mask.
        """
        q, k, v, mask, full, lse = ctx.saved_tensors
        b, h, n, d = q.shape
        grad = grad.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        delta = torch.empty((b * h, n), dtype=torch.float32, device=q.device)
        smb, smh, sm, sn = _mask_strides(mask)
        grid = (triton.cdiv(n, BACKWARD_TILE), b * h)
        _delta[grid](full, grad, delta, n, d, max(16, triton.next_power_of_2(d)), BACKWARD_TILE, num_warps=4)
        _backward_q[grid](q, k, v, mask, grad, lse, delta, dq, n, d, max(16, triton.next_power_of_2(d)), h,
                          smb, smh, sm, sn, CAUSAL, PRECISION_TERMS, BACKWARD_TILE,
                          num_warps=BACKWARD_WARPS, num_stages=2)
        _backward_kv[grid](q, k, v, mask, grad, lse, delta, dk, dv, n, d, max(16, triton.next_power_of_2(d)), h,
                           smb, smh, sm, sn, CAUSAL, PRECISION_TERMS, BACKWARD_TILE,
                           num_warps=BACKWARD_WARPS, num_stages=2)
        return dq, dk, dv, None


def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute causal or block-masked attention with first-order autograd.

    Args:
        inputs: Q, K, V tensors and their public Boolean attention mask.

    Returns:
        A BF16 CUDA tensor with the same shape as Q.
    """
    return _Attention.apply(inputs['q'], inputs['k'], inputs['v'], inputs['mask'])
