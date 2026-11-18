"""Speed-ups for training, chosen per GPU and model family: the attention kernel (FlashAttention 3 / 2, else SDPA) and
Liger Kernel (fused Triton RMSNorm / SwiGLU / GeGLU / RoPE, and a fused linear cross-entropy for sft.py).

  python3 accel.py check [--model <dir>]     what sft.py / grpo.py would pick on this machine, and whether it loads
  python3 accel.py fetch [--dest DIR]        download the FA3 / FA2 kernels for this torch into DIR (needs network once)

Attention (attn_implementation): FlashAttention 3 runs on Hopper (sm90) only; on Ampere (sm80) and newer non-Hopper
GPUs FlashAttention 2, if it loads; otherwise PyTorch SDPA. transformers 5 loads both from the flash-attn packages if
installed, else through the `kernels` package from the kernels-community/vllm-flash-attn3 and kernels-community/flash-attn2
builds. Without network those must be on disk: `fetch` copies them to /opt/kit-kernels (or $KIT_KERNELS), and
use_local_kernels() points the `kernels` package there (LOCAL_KERNELS) before anything loads.

Liger (liger_patch / fused_ce_loss): the layer kernels come from liger's apply_liger_kernel_to_<family> for the model's
config.model_type (qwen2, qwen3, llama, gemma3 / gemma3_text, granite, mistral, ...; not mistral3, i.e. Ministral 3, nor
granitemoehybrid); the loss is computed here with liger's fused linear cross-entropy on the language model's
last-layer outputs, so the full logits are never materialized (Gemma's final soft-capping and Granite's logit scaling
are applied). Liger's kernels are Triton: CUDA only.
"""
import argparse
import functools
import inspect
import os
import shutil
import sys
from pathlib import Path

KERNEL_DIR = os.environ.get("KIT_KERNELS", "/opt/kit-kernels")
# transformers 5.19's kernel fallbacks (modeling_flash_attention_utils.FLASH_ATTN_KERNEL_FALLBACK / _VERSIONS):
# implementation -> (hub repo, kernel version), and the dir name under KERNEL_DIR
FA_KERNELS = {"flash_attention_3": ("kernels-community/vllm-flash-attn3", 1),
              "flash_attention_2": ("kernels-community/flash-attn2", 3)}
ATTN_CHOICES = {"auto": None, "sdpa": "sdpa", "fa2": "flash_attention_2", "fa3": "flash_attention_3"}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- attention

def use_local_kernels(kernel_dir=None):
    """Point the `kernels` package at the FA kernels under kernel_dir (if there): adds repo=path entries to
    LOCAL_KERNELS for those not already in it. Returns the repos it found."""
    root = Path(kernel_dir or KERNEL_DIR)
    cur = os.environ.get("LOCAL_KERNELS", "")
    have = {e.split("=", 1)[0] for e in cur.split(":") if "=" in e}
    found, add = [], []
    for repo, _ in FA_KERNELS.values():
        d = root / repo.split("/", 1)[1]
        if (d / "build").is_dir():
            found.append(repo)
            if repo not in have:
                add.append(f"{repo}={d}")
    if add:
        os.environ["LOCAL_KERNELS"] = ":".join(([cur] if cur else []) + add)
    return found


@functools.lru_cache(maxsize=None)
def fa_loads(impl):
    """True if transformers can run `impl` (flash_attention_3 / flash_attention_2) here: its flash-attn package is
    installed, or the `kernels` package loads its hub kernel (from LOCAL_KERNELS / the HF cache / the Hub)."""
    try:
        from transformers.utils import import_utils as iu
        if impl == "flash_attention_3" and iu.is_flash_attn_3_available():
            return True
        if impl == "flash_attention_2" and iu.is_flash_attn_2_available():
            return True
        if not iu.is_kernels_available():
            return False
        use_local_kernels()
        from kernels import get_kernel
        repo, version = FA_KERNELS[impl]
        k = get_kernel(repo, version=version)
        return hasattr(k, "flash_attn_varlen_func")
    except Exception as e:   # no build for this torch / GPU, offline without the files, ...
        log(f"attention: {impl} does not load ({type(e).__name__}: {str(e)[:200]})")
        return False


def gpu_capability():
    import torch
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def supports_flash(model_dir):
    """Whether the model's transformers class has a FlashAttention path (all kit families do)."""
    if not model_dir:
        return True
    try:
        import chat
        return bool(getattr(chat.model_class(model_dir), "_supports_flash_attn", False))
    except Exception:
        return True


def attn_implementation(model_dir=None, choice="auto", capability=None, loads=None):
    """(attn_implementation for from_pretrained, reason) for `--attn choice` on this GPU: auto takes FA3 on Hopper
    (sm90), FA2 on sm80+ (A100, L40, ...), else SDPA, each only if it loads. An explicit fa3 / fa2 that cannot run
    here exits with the reason. capability / loads are for tests (default: the current GPU, fa_loads)."""
    if choice not in ATTN_CHOICES:
        raise SystemExit(f"--attn {choice!r}: choose one of {', '.join(ATTN_CHOICES)}")
    if choice == "sdpa":
        return "sdpa", "requested"
    cap = gpu_capability() if capability is None else capability
    loads = loads or fa_loads
    gpu = f"sm{cap[0]}{cap[1]}" if cap else "no GPU"
    can = {"flash_attention_3": bool(cap) and cap[0] == 9,       # Hopper only (FA3's kernels are sm90a)
           "flash_attention_2": bool(cap) and cap[0] >= 8}       # Ampere and newer
    why = {"flash_attention_3": "needs a Hopper GPU (sm90)", "flash_attention_2": "needs an Ampere or newer GPU (sm80+)"}
    flash_ok = supports_flash(model_dir)
    if choice != "auto":
        impl = ATTN_CHOICES[choice]
        if not flash_ok:
            raise SystemExit(f"--attn {choice}: this model class has no FlashAttention support in transformers")
        if not can[impl]:
            raise SystemExit(f"--attn {choice}: {impl} {why[impl]}; this GPU is {gpu}")
        if not loads(impl):
            raise SystemExit(f"--attn {choice}: {impl} does not load here (no flash-attn package and no kernel under "
                             f"{KERNEL_DIR}: `python3 accel.py fetch` downloads it)")
        return impl, f"requested, {gpu}"
    if not flash_ok:
        return "sdpa", "auto: the model class has no FlashAttention path"
    for impl in ("flash_attention_3", "flash_attention_2"):
        if can[impl] and loads(impl):
            return impl, f"auto, {gpu}"
    tried = [i for i in ("flash_attention_3", "flash_attention_2") if can[i]]
    return "sdpa", f"auto, {gpu}" + (f": {' / '.join(tried)} did not load" if tried else "")


# ---------------------------------------------------------------- Liger

def liger_available():
    try:
        import liger_kernel  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def model_type_of(model_or_config):
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "model_type", None)


def liger_apply_fn(model_type):
    """liger's apply_liger_kernel_to_<family> for a config.model_type, or None (not supported, or liger missing)."""
    try:
        from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    except Exception:
        return None
    return MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)


LIGER_OFF = {"cross_entropy": False, "fused_linear_cross_entropy": False}   # the loss stays ours (sft) / TRL's (grpo)


def liger_patch(model, enabled=True):
    """Apply liger's layer kernels to a loaded model instance (before PEFT wraps it). Returns (the apply function's
    name or None, a note for the log): None if Liger is off, missing, or has nothing for this model type. The kernels
    replace module forwards in place: parameter names, PEFT targets and state dicts are unchanged."""
    if not enabled:
        return None, "off (--no-liger)"
    if not liger_available():
        return None, "not available (liger-kernel not installed or no CUDA)"
    mt = model_type_of(model)
    fn = liger_apply_fn(mt)
    if fn is None:
        return None, f"no layer kernels for model_type {mt}: the model's own layers"
    params = inspect.signature(fn).parameters
    fn(model=model, **{k: v for k, v in LIGER_OFF.items() if k in params})
    return fn.__name__, f"layer kernels {fn.__name__}"


def lm_parts(model):
    """(backbone, lm_head, softcap, logit divisor) of a causal LM or an image-text model (a PEFT model is unwrapped):
    logits = softcap * tanh(lm_head(backbone(...).last_hidden_state) / divisor / softcap)."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(m, m.base_model_prefix, None) or m.model
    cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    return (backbone, m.get_output_embeddings(), getattr(cfg, "final_logit_softcapping", None),
            float(getattr(cfg, "logits_scaling", None) or 1.0))


@functools.lru_cache(maxsize=None)
def _flce(softcap):
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    import torch
    return LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="sum", softcap=softcap,
                                            accum_dtype=torch.float32)


def fused_ce_loss(model, input_ids, attention_mask, labels, n_tgt, dtype=None):
    """sum over target tokens of -log p(next token) / n_tgt (labels -100 are ignored; position t predicts t+1), as
    sft.py's sliced loss computes it, without materializing the logits: the backbone's last-layer outputs and the LM
    head's weight go into liger's fused linear cross-entropy in `dtype` (bf16 by default, as under autocast) with
    fp32 accumulation."""
    import torch
    backbone, head, softcap, div = lm_parts(model)
    if hasattr(head, "lora_A"):   # a LoRA on the LM head: the fused path would read only its base weight
        return sliced_ce_loss(model, input_ids, attention_mask, labels, n_tgt)
    dtype = dtype or torch.bfloat16
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        h = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = h[:, :-1].reshape(-1, h.size(-1)).to(dtype)
    tgt = labels[:, 1:].reshape(-1)
    keep = tgt != -100   # only target positions go through the head: the prompt and padding cost nothing
    h, tgt = h[keep], tgt[keep]
    w = head.weight.to(dtype)
    if div != 1.0:
        w = w / div
    bias = None if getattr(head, "bias", None) is None else head.bias.to(dtype) / div
    return _flce(softcap)(w, h, tgt, bias) / n_tgt


def sliced_ce_loss(model, input_ids, attention_mask, labels, n_tgt):
    """The same loss from the model's full logits, in fp32 a slice of 4096 positions at a time (a whole micro-batch of
    fp32 logits is micro-tokens x vocab x 4 bytes: 16 GB for Gemma 3's 262k vocabulary at 16k tokens): the fallback
    when Liger is off or unavailable."""
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    flat, tgt = logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1)
    return sum(torch.nn.functional.cross_entropy(flat[k:k + 4096].float(), tgt[k:k + 4096], ignore_index=-100,
                                                 reduction="sum") for k in range(0, flat.size(0), 4096)) / n_tgt


def grpo_liger_options(model_dir, enabled=True):
    """GRPOConfig overrides that turn on TRL's Liger support for this model, and a log line. TRL 1.15 applies only
    liger's layer kernels (it computes log-probs with its own chunked LM head and has no Liger GRPO loss), so the
    policy_loss() hook, LoRA, --bf16 and the vLLM weight sync see the same parameters and loss path as without it."""
    import chat
    if not enabled:
        return {}, "Liger: off (--no-liger)"
    cls = chat.model_class(model_dir).__name__
    if not liger_available():
        return {}, f"Liger: not available (liger-kernel not installed or no CUDA) for {cls}"
    from transformers import AutoConfig
    mt = AutoConfig.from_pretrained(model_dir).model_type
    fn = liger_apply_fn(mt)
    if fn is None:
        return {}, f"Liger: no layer kernels for {cls} (model_type {mt}): off"
    return ({"use_liger_kernel": True, "liger_kernel_config": dict(LIGER_OFF)},
            f"Liger: on for {cls} ({fn.__name__}: layer kernels; loss: TRL's chunked LM head)")


# ---------------------------------------------------------------- image build / dev tools

def fetch_kernels(dest, impls=tuple(FA_KERNELS)):
    """Download each FA kernel's build for this torch / CUDA and copy it to dest/<name> (what use_local_kernels reads).
    kernels 0.17's own HfApi sends an empty user-agent part that huggingface_hub 2.4 / httpx reject, so a plain HfApi is
    patched in; the trust check is skipped for these two kernels-community repos (both are trusted publishers)."""
    import kernels.hf_hub as H
    import kernels.install as I
    from huggingface_hub import HfApi
    from kernels import __version__ as kv
    api = lambda user_agent=None: HfApi(library_name="kernels", library_version=kv, user_agent="posttrain-kit")  # noqa: E731
    H._get_hf_api = I._get_hf_api = api
    for impl in impls:
        repo, version = FA_KERNELS[impl]
        variant = Path(I.install_kernel(repo, version=version, trust_remote_code=[repo]))
        out = Path(dest) / repo.split("/", 1)[1] / "build" / variant.name
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(variant, out, symlinks=False)
        log(f"{impl}: {repo} v{version} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--model", default=None)
    c.add_argument("--attn", default="auto", choices=list(ATTN_CHOICES))
    f = sub.add_parser("fetch")
    f.add_argument("--dest", default=KERNEL_DIR)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch_kernels(a.dest)
        return
    impl, why = attn_implementation(a.model, a.attn)
    log(f"attention: {impl} ({why})")
    for i in FA_KERNELS:
        log(f"  {i}: {'loads' if fa_loads(i) else 'does not load'}")
    log(f"liger: {'available' if liger_available() else 'not available'}")
    if a.model:
        from transformers import AutoConfig
        mt = AutoConfig.from_pretrained(a.model).model_type
        fn = liger_apply_fn(mt)
        log(f"  {mt}: {fn.__name__ if fn else 'no layer kernels'}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
