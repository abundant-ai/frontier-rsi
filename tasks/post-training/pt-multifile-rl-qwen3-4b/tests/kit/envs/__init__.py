"""Multi-turn text environments. Each module defines:

    NAME, DIFFICULTIES
    generate(rng, difficulty) -> params (a JSON-able dict: one episode)
    class Env(params): reset() -> first observation (str); step(action_text) -> (observation, reward, done)
    HAS_GENERATOR (optional, default True): False for an environment whose episodes come from data, not generate()

The protocol is plain text: the model answers each observation with one line `<VERB>: <argument>` (the module's
docstring lists its verbs); anything it writes before that line is free reasoning. (multifile_repair uses tool calls
instead: see its docstring.) Rewards are given by the
environment (usually 1 on success at the end, 0 otherwise).
"""
import importlib

ENVS = ["multifile_repair"]


def load(name):
    if name not in ENVS:
        raise KeyError(f"unknown env {name!r}; known: {', '.join(ENVS)}")
    return importlib.import_module(f"envs.{name}")


def parse_action(text, verbs):
    """The last line of the form `VERB: arg` with VERB in verbs -> (VERB, arg), else (None, None)."""
    import re
    found = None
    for m in re.finditer(r"(?m)^\s*([A-Z]+)\s*:\s*(.+?)\s*$", text or ""):
        if m.group(1) in verbs:
            found = (m.group(1), m.group(2))
    return found or (None, None)
