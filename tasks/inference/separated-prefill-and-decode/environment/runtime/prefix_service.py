"""Real frozen-transformer execution for bounded shared-prefix KV reuse."""
import gc
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from policy import CandidateError, call


@dataclass
class Entry:
    tokens: tuple
    tensors: tuple
    used: int

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for pair in self.tensors for t in pair)


def frozen(device="cuda", smoke=False, assets="/opt/assets/model"):
    if smoke:
        torch.manual_seed(19)
        config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=4096)
        model = Qwen2ForCausalLM(config).float().to(device)
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            assets, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="sdpa").to(device)
    return model.eval().requires_grad_(False), tokenizer


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def token_bytes(model):
    config = model.config
    return (2 * config.num_hidden_layers * config.num_key_value_heads *
            (config.hidden_size // config.num_attention_heads) * next(model.parameters()).element_size())


def clone_prefix(cache, length):
    if length == 0:
        return None
    layers = cache.layers if isinstance(cache, DynamicCache) else None
    pairs = ((layer.keys, layer.values) for layer in layers) if layers is not None else cache
    return tuple((k[..., :length, :].clone().contiguous(), v[..., :length, :].clone().contiguous()) for k, v in pairs)


def cache_object(tensors):
    return DynamicCache(ddp_cache_data=tensors) if tensors is not None else None


def requests(tokenizer, seed, quick=False, smoke=False):
    """Token-exact shared documents, unrelated distractors and late divergence.

    Generated text workloads avoid network datasets. Correctness compares each
    real frozen-model output to an independently executed uncached request.
    """
    rng = random.Random(seed)
    count = 6 if quick or smoke else 48
    groups = 3 if quick or smoke else 10
    prefixes = []
    for group in range(groups):
        if smoke:
            prefix = [rng.randrange(4, 120) for _ in range(24 + 8 * (group % 3))]
        else:
            text = f"Reference handbook for department {rng.randrange(10**8)}.\n"
            for page in range(180):
                text += (f"Record {page}: region {rng.randrange(10000)}, retention {rng.randrange(1,90)} days. "
                         "Keep the listed records distinct. Use only this handbook when answering.\n")
            length = (512, 1024, 1536)[group % 3]
            prefix = tokenizer.encode(text, add_special_tokens=False)[:length]
        prefixes.append(prefix)
    result = []
    for index in range(count):
        group = (index // 2) % groups if index % 7 else rng.randrange(groups)
        prefix = list(prefixes[group])
        # Some requests share nearly all of a prefix but deliberately differ
        # near the end. A cache policy must validate exact token identity.
        if index % 11 == 10:
            prefix[-3] = (prefix[-3] + 1) % (128 if smoke else tokenizer.vocab_size)
        suffix = ([4 + index % 100, 5, 6, 7] if smoke else
                  tokenizer.encode(f"\nQuestion {index}: summarize the handbook in one sentence. Answer:", add_special_tokens=False))
        result.append(tuple(prefix + suffix))
    return result


def checked_plan(policy, tokens, entries, budget, bytes_per_token):
    summary = tuple((entry.tokens, entry.nbytes, entry.used) for entry in entries)
    result = call(policy, tokens, summary, budget, bytes_per_token)
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise CandidateError("plan must return (source_index, reuse_tokens, keep_indices, store_tokens)")
    source, reuse, keep, store = result
    if any(type(value) is not int for value in (source, reuse, store)):
        raise CandidateError("source, reuse and store must be integers")
    if source == -1:
        if reuse != 0:
            raise CandidateError("a cache miss must reuse zero tokens")
    elif not 0 <= source < len(entries):
        raise CandidateError("cache source index is out of range")
    elif not 0 < reuse < len(tokens) or reuse > len(entries[source].tokens):
        raise CandidateError("reuse length is out of range; at least one prompt token must be evaluated")
    elif entries[source].tokens[:reuse] != tokens[:reuse]:
        raise CandidateError("cache source does not match the exact input prefix")
    if not isinstance(keep, (list, tuple)) or any(type(i) is not int or not 0 <= i < len(entries) for i in keep):
        raise CandidateError("keep indices are invalid")
    if len(set(keep)) != len(keep) or not 0 <= store <= len(tokens):
        raise CandidateError("duplicate keep index or invalid store length")
    if sum(entries[i].nbytes for i in keep) + store * bytes_per_token > budget:
        raise CandidateError("persistent KV storage exceeds the task budget")
    return source, reuse, keep, store


@torch.inference_mode()
def execute(model, tokens, past, output_tokens, device):
    consumed = 0 if past is None else past[0][0].shape[-2]
    ids = torch.tensor(tokens[consumed:], dtype=torch.long, device=device)[None, :]
    output = model(ids, past_key_values=cache_object(past), use_cache=True, logits_to_keep=1)
    return output


@torch.inference_mode()
def generate(model, output, count, device):
    generated = []
    for index in range(count):
        next_id = output.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_id)
        if index + 1 < count:
            output = model(next_id, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
    return torch.cat(generated, -1).detach().cpu().tolist()[0]


def run_trace(model, policy, prompts, device="cuda", output_tokens=16, budget_bytes=192*1024**2):
    entries = []
    results = []
    reused = 0
    peak_persistent = 0
    bytes_per_token = token_bytes(model)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    began = time.perf_counter()
    for index, tokens in enumerate(prompts):
        source, reuse, keep, store = checked_plan(policy, tokens, entries, budget_bytes, bytes_per_token)
        past = clone_prefix(entries[source].tensors, reuse) if reuse else None
        output = execute(model, tokens, past, output_tokens, device)
        # Persist only model-produced prompt KV. Decode state evolves in a
        # separate cache object and cannot modify a shared stored prefix.
        saved = clone_prefix(output.past_key_values, store) if store else None
        entries = [entries[i] for i in keep]
        if store:
            entries.append(Entry(tokens[:store], saved, index))
        peak_persistent = max(peak_persistent, sum(entry.nbytes for entry in entries))
        results.append(generate(model, output, output_tokens, device))
        reused += reuse
        del output, past, saved
    sync(device)
    elapsed = time.perf_counter() - began
    peak_gpu = torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0
    del entries
    gc.collect()
    return {"outputs": results, "elapsed_seconds": elapsed, "reused_tokens": reused,
            "input_tokens": sum(map(len, prompts)), "output_tokens": len(prompts)*output_tokens,
            "peak_persistent_bytes": peak_persistent, "peak_gpu_bytes": peak_gpu}


def uncached(tokens, entries, budget, bytes_per_token):
    return -1, 0, (), 0


def evaluate(model, policy, prompts, device, quick=False, budget_bytes=192*1024**2):
    count = 4 if quick else 16
    # The warmup covers fixed model initialization. Candidate policy, cache
    # matching/copies, prefills and all decode operations remain timed.
    run_trace(model, uncached, prompts[:1], device, 2, budget_bytes)
    reference = run_trace(model, uncached, prompts, device, count, budget_bytes)
    measurements = []
    for _ in range(1 if quick else 3):
        measured = run_trace(model, policy, prompts, device, count, budget_bytes)
        if measured.pop("outputs") != reference["outputs"]:
            raise CandidateError("cached generation differs from independently evaluated uncached outputs")
        if measured["peak_persistent_bytes"] > budget_bytes:
            raise CandidateError("persistent KV budget exceeded")
        if measured["peak_gpu_bytes"] > 16 * 1024**3:
            raise CandidateError("total allocated GPU memory exceeds 16 GiB")
        measurements.append(measured)
    return {"latency_seconds": statistics.median(m["elapsed_seconds"] for m in measurements),
            "reference_elapsed_seconds": reference["elapsed_seconds"], "measurements": measurements,
            "verified_output_tokens": len(prompts)*count}
