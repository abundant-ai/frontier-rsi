"""Multi-turn rollouts for the text environments in envs/ (and plain single-turn rows in the same batch).

A row with an "env" field is an episode: the environment's first observation is the prompt; the model answers, the
environment replies (as a new user turn, chat.turn_suffix_ids), and so on until the environment says done, MAX_TURNS pass, or the episode
uses ctx["max_completion"] tokens, or the prompt plus the episode would pass CONTEXT tokens. The environment's text is masked out of the loss (env_mask 0) and its reward is
the episode's reward (env_reward). Rows without "env" get one sampled turn and are scored by their checker.

Environment steps of one round run in parallel threads (ENV_WORKERS, default: the CPU count up to 32): a step that
runs code (multifile_repair's run_python, its grading) waits on a sandbox process.

run_episodes() is also what envs/evaluate_env.py uses (greedy), so training and evaluation see the same protocol.
Change anything: temperature per turn, a greedy share (see greedy_mix.py), shaping (e.g. a small cost per turn),
how many turns, what the model sees ...
"""
import os
from concurrent.futures import ThreadPoolExecutor

from vllm import SamplingParams

import envs
import chat

MAX_TURNS = 64        # model replies per episode
TURN_TOKENS = 4096    # tokens per reply
CONTEXT = 32768       # prompt + every reply + every environment turn
TEMPERATURE = 1.0
ENV_WORKERS = int(os.environ.get("ENV_WORKERS", min(32, os.cpu_count() or 4)))
_POOL = None


def _step_all(eps, pending):
    """env.step(text) for every (i, text) in pending, in parallel; results in the same order."""
    global _POOL
    if len(pending) <= 1 or ENV_WORKERS <= 1:
        return [eps[i].step(text) for i, text in pending]
    if _POOL is None:
        _POOL = ThreadPoolExecutor(ENV_WORKERS)
    return list(_POOL.map(lambda it: eps[it[0]].step(it[1]), pending))


def run_episodes(llm, tok, rows, temperature, max_tokens, rng=None):
    n = len(rows)
    stop = chat.stop_ids(tok)
    eps = [envs.load(r["env"]).Env(r["params"]) if r.get("env") else None for r in rows]
    # a training row may bring its own first message ("train_prompt"); graded rows never have one
    first = [(r.get("train_prompt") or e.reset()) if e else r["prompt"] for e, r in zip(eps, rows)]
    prompt_ids = [chat.prompt_ids(tok, [{"role": "user", "content": o}]) for o in first]
    budget = [min(max_tokens, CONTEXT - len(p)) for p in prompt_ids]   # an episode never passes the context
    comp, mask, lps = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    reward, turns, live, cut = [None if e is None else 0.0 for e in eps], [0] * n, set(range(n)), [False] * n
    for _ in range(MAX_TURNS):
        idx = sorted(i for i in live if budget[i] - 1 - len(comp[i]) > 8)   # 1 token kept for the closing eos
        live &= set(idx)
        if not idx:
            break
        # an explicit seed per request: identical requests (a group's copies of one episode) are otherwise sampled
        # identically by the trainer's vLLM engine, and a group whose rollouts agree teaches GRPO nothing
        params = [SamplingParams(temperature=temperature, max_tokens=min(TURN_TOKENS if eps[i] else max_tokens,
                                 budget[i] - 1 - len(comp[i])), stop_token_ids=stop, logprobs=0,
                                 seed=rng.randrange(2 ** 31) if rng else None) for i in idx]
        outs = llm.generate([{"prompt_token_ids": prompt_ids[i] + comp[i]} for i in idx], sampling_params=params, use_tqdm=False)
        pending, last = [], {}
        for i, o in zip(idx, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            comp[i] += ids; mask[i] += [1] * len(ids); lps[i] += [d[t].logprob for d, t in zip(c.logprobs, ids)]
            turns[i] += 1
            if eps[i] is None or c.finish_reason == "length":
                cut[i] = c.finish_reason == "length"
                live.discard(i); continue
            pending.append((i, chat.final_text(tok.decode(ids, skip_special_tokens=True))))
            last[i] = ids[-1] if ids else None
        for (i, _), (obs, r, done) in zip(pending, _step_all(eps, pending)):
            reward[i] = max(reward[i], r)
            if done:
                live.discard(i); continue
            sfx = chat.turn_suffix_ids(tok, obs, last[i])
            comp[i] += sfx; mask[i] += [0] * len(sfx); lps[i] += [0.0] * len(sfx)
    for i in range(n):   # close every episode the model did not cut off mid-reply (the environment ended it, or it ran
        # out of turns or tokens between replies) with the eos, so the trainer reads it as finished, not truncated
        if eps[i] is not None and not cut[i] and (not comp[i] or comp[i][-1] != tok.eos_token_id):
            comp[i].append(tok.eos_token_id); mask[i].append(0); lps[i].append(0.0)
    return {"prompt_ids": prompt_ids, "completion_ids": comp, "logprobs": lps, "env_mask": mask, "env_reward": reward,
            "turns": turns}


def rollout(prompts, rows, trainer, ctx):
    out = run_episodes(ctx["llm"], ctx["tok"], rows, ctx.get("temperature", TEMPERATURE), ctx["max_completion"], ctx["rng"])
    out.pop("turns")
    return out
