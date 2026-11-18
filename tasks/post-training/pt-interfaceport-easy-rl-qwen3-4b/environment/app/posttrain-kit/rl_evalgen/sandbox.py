"""Run Python code in a locked-down child process: a project's files (run_python, and the hidden tests) and single
functions.

  * `screen` rejects the source before anything runs: unparseable code, imports outside `ALLOWED_IMPORTS`, the builtins
    that reach outside the program (open, eval, exec, getattr, ...), dunder attributes and frame or code attributes.
  * The child is a fresh `python -S -s -P` with an empty environment (only a fixed hash seed, so iterating over a set
    of strings repeats exactly), in its own session, its output in a size-limited temporary file. Before the code
    runs, it sets resource limits (CPU, memory, file size, processes), drops from root to `nobody` when it is root,
    leaves the network namespace when it can, installs an audit hook that refuses files, sockets, processes and new
    imports, and replaces the builtins.
  * When functions are called on inputs, the child gets the inputs only, never the expected outputs. It reports what
    each function returned, in plain data (`encode` refuses any other type, so an object whose `__eq__` always says
    True cannot get through); the parent compares (`same`). Prints are discarded, so printing cannot fake a result.

A wall-clock limit in the parent ends the whole process group, and on macOS, where RLIMIT_AS does nothing, the parent
also watches the child's memory. A call that crashes, times out or returns something `encode` refuses counts as a
wrong result. A sandbox that cannot be set up (the child cannot drop root, say) fails every run, so `selftest`
notices it.

Inputs are lists of arguments, encoded with `encode`. For a single function (`run_function`), an argument may
also be `expr("<python expression>")`: an expression evaluated in the child, so a test with a very large input
stays small in the data. Projects (`call_project`) take encoded values only.

Python's own restrictions are not a security boundary on their own; the operating-system limits above are what hold
if the code gets past them.

Projects (`run_project`, `call_project`): several flat .py files that import each other, run in the same kind of child.
`run_project` runs one file as __main__ and returns what it printed and the traceback (project frames only);
`call_project` calls functions on inputs and returns the outcomes.

Standard library only, and no other imports from this package, so it can be copied as one file.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

ALLOWED_IMPORTS = {
    "bisect", "collections", "copy", "dataclasses", "datetime", "decimal", "fractions", "functools", "heapq",
    "itertools", "math", "operator", "re", "statistics", "string", "sys", "typing",
}
# Third-party libraries a job may ask for, with the submodules loaded before the lockdown (afterwards no module can be
# imported, so only these are reachable: `from scipy import stats` works, scipy.io does not exist).
LIBRARIES = {
    "numpy": ["numpy", "numpy.linalg", "numpy.random", "numpy.fft", "numpy.lib", "numpy.lib.stride_tricks"],
    "scipy": ["scipy", "scipy.special", "scipy.stats", "scipy.linalg", "scipy.signal", "scipy.optimize", "scipy.ndimage",
              "scipy.spatial", "scipy.spatial.distance", "scipy.interpolate", "scipy.integrate", "scipy.sparse",
              "scipy.sparse.csgraph"],
    "torch": ["torch", "torch.nn", "torch.nn.functional"],
    "pandas": ["pandas"],
}
LIBRARY_NEEDS = {"scipy": ["numpy"], "torch": ["numpy"], "pandas": ["numpy"]}
# pandas methods that evaluate a string of code (DataFrame.eval / query, pd.eval): refused by the screen
PANDAS_DENIED_ATTRIBUTES = {"eval", "query"}
# Libraries import some of their own modules on first use (numpy's reductions load numpy._core._methods from C, for
# example). After the lockdown that would be a refused import, so each child first runs this code, which loads them.
LIBRARY_WARMUP = {
    "numpy": "import numpy as np\n"
             "a = np.arange(12.0).reshape(3, 4)\n"
             "a.max(); a.min(); a.mean(axis=0); a.std(axis=1); a.var(); a.sum(); a.prod(); a.argmax(); a.cumsum(); a.any(); a.all()\n"
             "np.unique(np.array([3, 1, 3])); np.sort(a, axis=0); np.argsort(-a, kind='stable'); np.searchsorted([1, 2], 1.5)\n"
             "np.median(a); np.percentile(a, 50); np.clip(a, 1, 2); np.round(a, 1); np.isclose(a, a); np.allclose(a, a)\n"
             "np.add.at(np.zeros(3), [0, 1], 1); np.bincount([0, 1]); np.minimum.accumulate(a[0]); np.diff(a); np.where(a > 1)\n"
             "np.concatenate(([0.0], a[0])); np.stack([a, a]); np.linalg.norm(a); np.linalg.inv(np.eye(2)); np.linalg.solve(np.eye(2), [1.0, 2.0])\n"
             "np.lib.stride_tricks.sliding_window_view(a[0], 2); np.convolve([1.0, 2.0], [1.0]); np.correlate([1.0, 2.0], [1.0])\n"
             "np.histogram(a); np.interp([0.5], [0, 1], [0, 1]); np.einsum('ij->j', a); repr(a); str(a); np.array2string(a)\n"
             "np.random.default_rng(0).random(2); np.fft.fft([1.0, 2.0]); np.nanmax(a); np.isnan(a); np.tile(a, 2); np.pad(a, 1)\n",
    "scipy": "import numpy as np, scipy\n"
             "x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])\n"
             "scipy.stats.rankdata(x); scipy.stats.zscore(x); scipy.stats.gmean(x); scipy.stats.pearsonr(x, x[::-1]); scipy.stats.spearmanr(x, x[::-1])\n"
             "scipy.stats.binom.pmf(1, 3, 0.5); scipy.stats.norm.cdf(0.5); scipy.stats.describe(x); scipy.stats.mode(x); scipy.stats.linregress(x, x)\n"
             "scipy.special.erf(0.5); scipy.special.gammaln(3.0); scipy.special.comb(5, 2); scipy.special.expit(0.0); scipy.special.logsumexp(x)\n"
             "scipy.signal.medfilt(x, 3); scipy.signal.find_peaks(x); scipy.signal.convolve2d(np.eye(3), np.ones((2, 2)), mode='valid'); scipy.signal.lfilter([1.0], [1.0, -0.5], x)\n"
             "scipy.linalg.solve(np.eye(2), [1.0, 2.0]); scipy.linalg.det(np.eye(2)); scipy.linalg.lu_factor(np.eye(2)); scipy.linalg.expm(np.eye(2))\n"
             "scipy.spatial.distance.cdist(np.eye(2), np.eye(2)); scipy.spatial.distance.pdist(np.eye(3)); scipy.spatial.cKDTree(np.eye(3)).query(np.eye(3)[0])\n"
             "scipy.integrate.trapezoid(x); scipy.integrate.cumulative_trapezoid(x); scipy.interpolate.interp1d([0, 1], [0, 1])(0.5)\n"
             "scipy.ndimage.label(np.eye(3)); scipy.ndimage.uniform_filter(np.eye(3), 2); scipy.ndimage.binary_dilation(np.eye(3))\n"
             "scipy.optimize.linear_sum_assignment(np.eye(3)); scipy.optimize.brentq(lambda t: t - 0.5, 0, 1); scipy.optimize.minimize_scalar(lambda t: (t - 1) ** 2)\n"
             "scipy.sparse.csgraph.shortest_path(np.eye(3)); scipy.sparse.csgraph.connected_components(np.eye(3)); scipy.sparse.csr_matrix(np.eye(2)).toarray()\n",
    "torch": "import torch\nimport torch.nn.functional as F\n"
             "x = torch.randn(3, 4)\n"
             "x.max(); x.min(dim=0); x.mean(dim=1); x.std(); x.var(); x.sum(); x.argmax(dim=1); x.cumsum(dim=0); x.softmax(dim=-1); x.sort(dim=1); x.topk(2)\n"
             "F.softmax(x, dim=-1); F.log_softmax(x, dim=-1); F.relu(x); F.gelu(x); F.cross_entropy(x, torch.tensor([0, 1, 2])); F.nll_loss(F.log_softmax(x, -1), torch.tensor([0, 1, 2]))\n"
             "F.one_hot(torch.tensor([0, 2]), 3); F.pad(x, (1, 1)); F.layer_norm(x, (4,)); F.normalize(x, dim=1); F.cosine_similarity(x, x, dim=1); F.mse_loss(x, x)\n"
             "F.conv1d(x[None], torch.randn(2, 3, 2)); F.avg_pool1d(x[None], 2); F.embedding(torch.tensor([0, 1]), x); F.kl_div(F.log_softmax(x, -1), F.softmax(x, -1), reduction='batchmean')\n"
             "torch.einsum('ij,kj->ik', x, x); x @ x.T; torch.tril(torch.ones(3, 3)); x.masked_fill(x > 0, float('-inf')); torch.where(x > 0, x, 0.0); torch.gather(x, 1, torch.tensor([[0], [1], [2]]))\n"
             "torch.cat([x, x]); torch.stack([x, x]); torch.linalg.norm(x); torch.cdist(x, x); torch.nn.utils.rnn.pad_sequence([x, x[:2]], batch_first=True); repr(x); str(x)\n"
             "torch.manual_seed(0); torch.rand(2); torch.arange(5).float().exp().log(); torch.allclose(x, x); torch.isclose(x, x); x.tolist(); x.numpy()\n",
    "pandas": "import numpy as np, pandas as pd\n"
              "df = pd.DataFrame({'g': ['a', 'b', 'a', 'c'], 'v': [1, 2, 3, 4], 'w': [0.5, 1.5, 2.5, 3.5], 'ok': [True, False, True, True]})\n"
              "s = df['v']; t = df['w']; g = df.groupby('g')\n"
              "s.sum(); s.mean(); s.median(); s.min(); s.max(); s.std(); s.var(); s.count(); s.nunique(); s.prod(); s.quantile(0.5); s.mode(); s.abs(); s.round(1)\n"
              "s.cumsum(); s.cummax(); s.cummin(); s.cumprod(); s.diff(); s.shift(1); s.pct_change(); s.rank(); s.rank(method='min'); s.rank(method='dense', ascending=False)\n"
              "s.idxmax(); s.idxmin(); s.nlargest(2); s.nsmallest(2); s.value_counts(); s.unique(); s.isin([1, 2]); s.between(1, 3); s.clip(1, 3); s.where(s > 1, 0); s.mask(s > 1, 0)\n"
              "s.sort_values(); s.sort_index(); s.astype(float); s.map({1: 2}); s.apply(lambda x: x); s.fillna(0); s.dropna(); s.duplicated(); s.drop_duplicates(); s.head(2); s.tail(2); s.to_list(); s.to_numpy(); s.to_dict()\n"
              "s.rolling(2).mean(); s.rolling(2, min_periods=1).sum(); s.rolling(2).max(); s.expanding().mean(); s.ewm(alpha=0.5).mean(); t.interpolate(); s.groupby(df['g']).cumsum()\n"
              "g['v'].sum(); g['v'].mean(); g['v'].count(); g.size(); g['v'].max(); g['v'].min(); g['v'].std(); g['v'].var(); g['v'].median(); g['v'].nunique(); g['v'].first(); g['v'].last()\n"
              "g['v'].cumsum(); g['v'].cumcount(); g['v'].rank(); g['v'].transform('mean'); g['v'].transform('std'); g['v'].shift(1); g['v'].diff(); g['v'].nlargest(1); g['v'].idxmax(); g['v'].agg(['sum', 'mean'])\n"
              "g.agg(total=('v', 'sum'), mean_w=('w', 'mean')); g.apply(lambda d: d['v'].sum(), include_groups=False); g['v'].apply(lambda x: x.sum()); df.groupby(['g', 'ok'])['v'].sum(); g.filter(lambda d: len(d) > 1); g.head(1)\n"
              "df.sort_values('v'); df.sort_values(['g', 'v'], ascending=[True, False]); df.sort_index(); df.reset_index(drop=True); df.set_index('g'); df.loc[df['v'] > 1]; df.iloc[1:3]; df.loc[0, 'v']; df.at[0, 'v']; df.iat[0, 1]\n"
              "df.drop_duplicates('g'); df.drop_duplicates('g', keep='last'); df.duplicated('g'); df.nlargest(2, 'v'); df.nsmallest(2, 'v'); df.assign(x=df['v'] * 2); df.rename(columns={'v': 'u'}); df.drop(columns=['w'])\n"
              "df.merge(df, on='g'); df.merge(df[['g', 'w']], on='g', how='left'); pd.merge(df, df, on='g', how='outer'); pd.concat([df, df]); pd.concat([s, t], axis=1); df.join(df.set_index('g')[['v']], on='g', rsuffix='_r')\n"
              "df.pivot_table(index='g', columns='ok', values='v', aggfunc='sum', fill_value=0); pd.crosstab(df['g'], df['ok']); df.melt(id_vars='g'); df.pivot_table(index='g', values='v', aggfunc='mean'); df.groupby('g')['v'].sum().unstack() if False else None\n"
              "df['g'].str.upper(); df['g'].str.len(); df['g'].str.contains('a'); df['g'].str.startswith('a'); df['g'].str.cat(sep=','); df['g'].str.split('a'); df['g'].str.strip(); df['g'].str.replace('a', 'b'); df['g'].astype('category')\n"
              "pd.cut(s, 2); pd.cut(s, [0, 2, 4], labels=False); pd.qcut(s, 2, labels=False); df.describe(); df.corr(numeric_only=True); df.cov(numeric_only=True); df.sum(numeric_only=True); df.mean(numeric_only=True); df.max(numeric_only=True); df.apply(lambda c: c, axis=0); df.apply(lambda r: r['v'], axis=1)\n"
              "df.isna(); df.notna(); df.fillna(0); df.dropna(); df.any(); df.all(); df.count(); df.nunique(); df.T; df.to_dict('records'); df.to_dict('list'); df.values; df.to_numpy(); df.copy(); df.equals(df); df.shape; df.columns.tolist(); df.index.tolist()\n"
              "repr(df); str(df); repr(s); str(s); df.to_string(); s.to_string(); df.head(); pd.Series([], dtype='str'); pd.Series([], dtype='int64'); pd.DataFrame(); pd.DataFrame({'a': []}); pd.Index([1, 2]); pd.RangeIndex(3); pd.isna(np.nan); pd.notna(1)\n"
              "df.groupby('g', sort=False)['v'].sum(); df.groupby('g', as_index=False)['v'].sum(); df.value_counts('g'); s.value_counts(normalize=True); s.value_counts(sort=False); df['g'].factorize(); pd.get_dummies(df['g']); df.explode('g'); df.where(df > 0) if False else None\n",
}


def lib_tops(libs) -> list:
    """The libraries a job loads: the ones asked for and those they need, in load order."""
    out = []
    for lib in libs:
        for t in LIBRARY_NEEDS.get(lib, []) + [lib]:
            if t not in out:
                out.append(t)
    return out


DENIED_NAMES = {
    "__builtins__", "__import__", "__loader__", "__spec__", "breakpoint", "compile", "delattr", "dir", "eval", "exec",
    "exit", "getattr", "globals", "help", "input", "locals", "memoryview", "open", "quit", "setattr", "vars",
}
ALLOWED_DUNDERS = {"__init__", "__name__"}  # super().__init__(), func.__name__
DENIED_ATTRIBUTES = {
    "ag_code", "ag_frame", "co_code", "cr_code", "cr_frame", "f_back", "f_builtins", "f_code", "f_globals", "f_locals",
    "func_code", "func_globals", "gi_code", "gi_frame", "gi_yieldfrom", "tb_frame", "tb_next",
}
MAX_CODE_CHARS = 100_000
MAX_OUTPUT_BYTES = 32 << 20


# --------------------------------------------------------------------------------------------------------------------
# Plain data: what crosses the process boundary, in both directions
# --------------------------------------------------------------------------------------------------------------------

# This text is also the first part of the child's program, so both sides encode the same way.
_CODEC = r'''
import json, math, sys

_KIND = {"b": "bool", "i": "int", "u": "int", "f": "float"}

def encode(x, depth=0):
    if depth > 200:
        raise ValueError("too deeply nested")
    t = type(x)
    if x is None or t is bool or t is int or t is str:
        return x
    if t is float:
        return x if math.isfinite(x) else {"__float__": repr(x)}
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, (np.ndarray, np.generic)):
        k = x.dtype.kind
        if k not in _KIND:
            raise TypeError("not plain data: values of type " + str(x.dtype))
        if isinstance(x, np.generic):
            return encode(x.item(), depth + 1)
        return {"__array__": [_KIND[k], [int(n) for n in x.shape], encode(np.ndarray.tolist(x), depth + 1)]}
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(x, (pd.Series, pd.DataFrame)):
        if isinstance(x, pd.Series):
            return {"__series__": [_pdkind(x), encode(x.index.tolist(), depth + 1), encode(x.tolist(), depth + 1)]}
        cols = [x.iloc[:, j] for j in range(x.shape[1])]
        return {"__frame__": [encode(x.columns.tolist(), depth + 1), encode(x.index.tolist(), depth + 1),
                              [[_pdkind(c), encode(c.tolist(), depth + 1)] for c in cols]]}
    th = sys.modules.get("torch")
    if th is not None and isinstance(x, th.Tensor):
        x = th.Tensor.detach(x).cpu()
        if x.is_complex() or x.dtype.is_complex:
            raise TypeError("not plain data: complex tensor")
        k = "bool" if x.dtype == th.bool else ("float" if x.is_floating_point() else "int")
        return {"__tensor__": [k, [int(n) for n in x.shape], encode(th.Tensor.tolist(x), depth + 1)]}
    if isinstance(x, list):
        return [encode(v, depth + 1) for v in list.__iter__(x)]
    if isinstance(x, tuple):
        return {"__tuple__": [encode(v, depth + 1) for v in tuple.__iter__(x)]}
    if isinstance(x, (set, frozenset)):
        return {"__set__": sorted((encode(v, depth + 1) for v in x), key=_order)}
    if isinstance(x, dict):
        return {"__dict__": sorted(([encode(k, depth + 1), encode(v, depth + 1)] for k, v in dict.items(x)),
                                   key=lambda kv: _order(kv[0]))}
    raise TypeError("not plain data: " + t.__name__)

def _order(v):
    return json.dumps(v, sort_keys=True)

def _pdkind(s):
    """A pandas column's kind: bool, int, float, or object (strings and other Python values)."""
    k = s.dtype.kind
    if k in _KIND:
        return _KIND[k]
    if k == "O":
        return "object"
    raise TypeError("not plain data: values of type " + str(s.dtype))

def _pdcol(kind, values):
    """A column's values as a pandas Series with a default index: the dtype from the kind; strings as pandas' string
    dtype (also when there are none)."""
    pd = sys.modules["pandas"]
    if kind == "object":
        dt = "str" if all(type(v) is str for v in values) else object
    else:
        dt = {"bool": "bool", "int": "int64", "float": "float64"}[kind]
    return pd.Series(values, dtype=dt)

def _pdindex(pd, idx, n):
    return None if idx == list(range(n)) else pd.Index(idx)

def decode(x):
    if isinstance(x, list):
        return [decode(v) for v in x]
    if isinstance(x, dict):
        if "__float__" in x:
            return float(x["__float__"])
        if "__tuple__" in x:
            return tuple(decode(v) for v in x["__tuple__"])
        if "__set__" in x:
            return {decode(v) for v in x["__set__"]}
        if "__dict__" in x:
            return {decode(k): decode(v) for k, v in x["__dict__"]}
        if "__array__" in x:
            np = sys.modules["numpy"]
            k, shape, data = x["__array__"]
            return np.array(decode(data), dtype={"bool": np.bool_, "int": np.int64, "float": np.float64}[k]).reshape(shape)
        if "__series__" in x:
            pd = sys.modules["pandas"]
            k, idx, vals = x["__series__"]
            s, idx = _pdcol(k, decode(vals)), decode(idx)
            if _pdindex(pd, idx, len(s)) is not None:
                s.index = _pdindex(pd, idx, len(s))
            return s
        if "__frame__" in x:
            pd = sys.modules["pandas"]
            cols, idx, data = x["__frame__"]
            cols, idx = decode(cols), decode(idx)
            df = pd.DataFrame({j: _pdcol(k, decode(v)) for j, (k, v) in enumerate(data)}, index=pd.RangeIndex(len(idx)))
            df.columns = pd.Index(cols, dtype=object) if cols else pd.RangeIndex(0)
            if _pdindex(pd, idx, len(idx)) is not None:
                df.index = _pdindex(pd, idx, len(idx))
            return df
        if "__tensor__" in x:
            th = sys.modules["torch"]
            k, shape, data = x["__tensor__"]
            return th.tensor(decode(data), dtype={"bool": th.bool, "int": th.int64, "float": th.float32}[k]).reshape(shape)
        raise ValueError("bad value")
    return x
'''


def _load_codec():
    ns = {}
    exec(_CODEC, ns)  # noqa: S102 - our own text
    return ns["encode"], ns["decode"]


encode, decode = _load_codec()


def expr(source: str) -> dict:
    """An argument computed in the child from a trusted expression (written into the data, never model text)."""
    return {"__expr__": source}


def same(a, b, rel_tol: float = 1e-9) -> bool:
    """Do two encoded values match? Numbers by value (5 == 5.0, floats within rel_tol), everything else strictly:
    True is not 1, a tuple is not a list, a string is not a number."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return abs(a - b) <= rel_tol * max(1.0, abs(a), abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, rel_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rel_tol) for k in a)
    return type(a) is type(b) and a == b


# --------------------------------------------------------------------------------------------------------------------
# Screening the source
# --------------------------------------------------------------------------------------------------------------------


def screen(code: str, modules=(), libs=()) -> str | None:
    """Why this source may not run, or None if it may. `modules`: a project's own module names, also importable;
    `libs`: third-party libraries the job loads (LIBRARIES keys)."""
    allowed = ALLOWED_IMPORTS | set(modules) | set(lib_tops(libs))
    if len(code) > MAX_CODE_CHARS:
        return "too long"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        return f"does not parse: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed:
                    return f"import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in allowed:
                return f"import from {node.module}"
            if any(a.name == "*" for a in node.names):
                return "import *"
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            return f"name {node.id}"
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a in DENIED_ATTRIBUTES or (a.startswith("__") and a not in ALLOWED_DUNDERS):
                return f"attribute {a}"
            if a in PANDAS_DENIED_ATTRIBUTES and "pandas" in allowed:
                return f"attribute {a} (evaluates a string of code)"
    return None


# --------------------------------------------------------------------------------------------------------------------
# The child
# --------------------------------------------------------------------------------------------------------------------

# Shared by both children: the allowed standard modules, the lockdown, and the builtins the code may use.
_LOCKDOWN = r'''
import os, sys

def _stdlib(libs=()):
    """Import the allowed modules (and the job's libraries) before the lockdown (afterwards, imports are refused)."""
    import bisect, collections, copy, dataclasses, datetime, decimal, fractions, functools, heapq, itertools, operator
    import re, statistics, string, types, typing, _strptime
    allowed = {"bisect": bisect, "collections": collections, "copy": copy, "dataclasses": dataclasses,
               "datetime": datetime, "decimal": decimal, "fractions": fractions, "functools": functools, "heapq": heapq,
               "itertools": itertools, "math": math, "operator": operator, "re": re, "statistics": statistics,
               "string": string, "typing": typing}
    allowed["sys"] = types.SimpleNamespace(maxsize=sys.maxsize, float_info=sys.float_info, version_info=sys.version_info,
                                           setrecursionlimit=lambda n: sys.setrecursionlimit(min(int(n), 20000)),
                                           getrecursionlimit=sys.getrecursionlimit,
                                           set_int_max_str_digits=lambda n: None)
    submodules = {"collections.abc": collections.abc}
    import importlib
    for name in libs:
        m = importlib.import_module(name)
        if "." in name:
            submodules[name] = m
        else:
            allowed[name] = m
    if "pandas" in allowed:                 # strings stored the same way with or without pyarrow installed
        allowed["pandas"].set_option("mode.string_storage", "python")
    for name in list(allowed):
        if name in _WARMUP:
            exec(_WARMUP[name], {})
    if "torch" in allowed:
        allowed["torch"].set_num_threads(1)
        try:
            allowed["torch"].set_num_interop_threads(1)
        except RuntimeError:
            pass
    return allowed, submodules

def _lockdown(job):
    sys.setrecursionlimit(5000)
    if os.geteuid() == 0:                   # as `nobody`: no access to root-only files
        if hasattr(os, "unshare"):
            try:
                os.unshare(os.CLONE_NEWNET)
            except OSError:
                pass
        os.setgroups([])
        os.setresgid(65534, 65534, 65534)
        os.setresuid(65534, 65534, 65534)
        if os.geteuid() == 0:
            os._exit(99)
    import resource
    cpu, mem, fsize = job["cpu_s"], job["memory_mb"] << 20, job["max_output"]
    for name, limit in (("RLIMIT_CPU", (cpu, cpu + 1)), ("RLIMIT_AS", (mem, mem)), ("RLIMIT_FSIZE", (fsize, fsize)),
                        ("RLIMIT_CORE", (0, 0)), ("RLIMIT_NPROC", (0, 0)), ("RLIMIT_NOFILE", (16, 16))):
        try:
            resource.setrlimit(getattr(resource, name), limit)
        except (ValueError, OSError):       # macOS refuses some of these; the wall-clock limit still holds
            pass

    denied = ("open", "socket.", "subprocess.", "os.", "ctypes.", "shutil.", "pty.", "mmap.", "fcntl.", "signal.",
              "sys.settrace", "sys.setprofile", "sys.addaudithook", "sys.unraisablehook", "cpython.", "import",
              "glob.", "tempfile.", "resource.", "syslog.", "urllib.", "ftplib.", "smtplib.", "http.", "webbrowser.")

    def hook(event, args):
        if event.startswith(denied):
            raise PermissionError(event)

    sys.addaudithook(hook)

def _std_import(allowed, submodules, name, fromlist):
    top = name.split(".")[0]
    if "." in name:
        m = submodules.get(name)
        if m is None and top in _WARMUP and top in allowed:   # a library's own module, already loaded by the library
            m = sys.modules.get(name)
        if m is None:
            raise ImportError(name)
        return m if fromlist else allowed[top]
    return allowed[top]

def _safe_builtins(importer, extra=()):
    safe = {k: __builtins__.__dict__[k] for k in (
        "abs all any ascii bin bool bytearray bytes callable chr classmethod complex dict divmod enumerate filter float "
        "format frozenset hasattr hash hex id int isinstance issubclass iter len list map max min next object oct ord "
        "pow print property range repr reversed round set slice sorted staticmethod str sum super tuple type zip "
        "__build_class__ Exception BaseException ArithmeticError AssertionError AttributeError IndexError KeyError "
        "LookupError NameError NotImplementedError OverflowError RecursionError RuntimeError StopIteration TypeError "
        "ValueError ZeroDivisionError NotImplemented Ellipsis True False None").split() + list(extra)}
    safe["__import__"] = importer
    return safe
'''

_CHILD = _CODEC + "\n_WARMUP = " + repr(LIBRARY_WARMUP) + "\n" + _LOCKDOWN + r'''
def main():
    job = json.load(sys.stdin)
    result = os.fdopen(os.dup(1), "w")           # the only way out; prints go to /dev/null
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)

    def arg(x):
        if isinstance(x, dict) and "__expr__" in x:
            return eval(x["__expr__"], {"__builtins__": {"range": range, "len": len, "abs": abs, "min": min, "max": max,
                                                         "sum": sum, "list": list, "str": str, "chr": chr, "ord": ord,
                                                         "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
                                                         "zip": zip, "tuple": tuple, "int": int}})
        return decode(x)

    sys.path.extend(job.get("lib_paths", []))
    allowed, submodules = _stdlib(job.get("libs", []))
    inputs = [[arg(a) for a in args] for args in job["inputs"]]
    _lockdown(job)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in allowed:
            raise ImportError(name)
        return _std_import(allowed, submodules, name, fromlist)

    env = {"__builtins__": _safe_builtins(guarded_import), "__name__": "solution"}

    out = {"outputs": None, "error": None}
    try:
        exec(compile(job["code"], "<solution>", "exec"), env)
        fn = env[job["func"]]
        outputs = []
        for args in inputs:
            outputs.append(encode(fn(*args)))
        out["outputs"] = outputs
    except BaseException as e:
        out["error"] = type(e).__name__ + ": " + str(e)[:200]
    result.write(json.dumps(out))
    result.flush()
    os._exit(0)

main()
'''

# A project: several files that import each other. Two jobs: "run" runs one file as __main__ and reports what it
# printed and its traceback (for the student); "call" calls functions on inputs and reports the outcomes (for grading).
_PROJECT_CHILD = _CODEC + "\n_WARMUP = " + repr(LIBRARY_WARMUP) + "\n" + _LOCKDOWN + r'''
class _Capture:
    """Where print goes: keeps the first `limit` characters and counts the rest."""
    def __init__(self, limit):
        self.parts, self.n, self.limit = [], 0, limit
    def write(self, s):
        s = str(s)
        if self.n < self.limit:
            self.parts.append(s[: self.limit - self.n])
        self.n += len(s)
        return len(s)
    def flush(self):
        pass
    def text(self):
        t = "".join(self.parts)
        return t + (f"\n[... {self.n - self.limit} more characters not shown]" if self.n > self.limit else "")

def main():
    job = json.load(sys.stdin)
    result = os.fdopen(os.dup(1), "w")           # the only way out
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    import linecache, traceback, types
    files = job["files"]                          # {"stats.py": source}
    stems = {f[:-3]: f for f in files}
    for f, src in files.items():                  # tracebacks show the lines without reading any file
        linecache.cache[f] = (len(src), None, [ln + "\n" for ln in src.split("\n")], f)
    sys.path.extend(job.get("lib_paths", []))
    allowed, submodules = _stdlib(job.get("libs", []))
    calls = [[[decode(a) for a in args] for args in c["inputs"]] for c in job.get("calls", [])]
    _lockdown(job)
    cap = _Capture(job["max_print"])
    sys.stdout = sys.stderr = cap

    modules = {}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if level:
            raise ImportError("relative imports are not allowed")
        if top in allowed:
            return _std_import(allowed, submodules, name, fromlist)
        if name in stems:
            return load(name)
        raise ImportError(f"No module named {name!r}: only the project's files and the allowed standard modules")

    safe = _safe_builtins(guarded_import, ["SystemExit"])

    def load(name, as_main=False):
        if not as_main and name in modules:
            return modules[name]
        m = types.ModuleType("__main__" if as_main else name)
        m.__dict__["__builtins__"] = safe
        if not as_main:
            modules[name] = m
        try:
            exec(compile(files[stems[name]], stems[name], "exec"), m.__dict__)
        except BaseException:
            modules.pop(name, None)
            raise
        return m

    def show(e):
        frames = [fr for fr in traceback.extract_tb(e.__traceback__) if fr.filename in files]
        head = "Traceback (most recent call last):\n" + "".join(traceback.format_list(frames)) if frames else ""
        return head + "".join(traceback.format_exception_only(type(e), e))

    out = {}
    if job["mode"] == "run":
        code, tb = 0, ""
        try:
            load(job["entry"][:-3], as_main=True)
        except SystemExit as e:
            if e.code is None or isinstance(e.code, int):
                code = e.code or 0
            else:
                code, tb = 1, str(e.code) + "\n"
        except BaseException as e:
            code, tb = 1, show(e)
        out = {"printed": cap.text(), "traceback": tb[:20000], "exit_code": code}
    else:
        results = []
        for c, inputs in zip(job["calls"], calls):
            modules.clear()                       # a fresh copy of the project for every function
            try:
                fn = getattr(load(c["module"]), c["func"])
            except BaseException as e:
                results.append({"error": show(e)[-500:]})
                continue
            got = []
            for args in inputs:
                try:
                    v = fn(*args)
                    try:
                        got.append([0, encode(v)])
                    except BaseException:
                        got.append([1, "not plain data: " + type(v).__name__])
                except BaseException as e:
                    got.append([1, type(e).__name__])
            results.append({"outcomes": got})
        out = {"results": results}
    result.write(json.dumps(out))
    result.flush()
    os._exit(0)

main()
'''


def _wait(p: subprocess.Popen, job: bytes, timeout_s: float, memory_mb: int) -> str | None:
    """Feed the child and wait for it; why it had to be stopped, or None. macOS ignores RLIMIT_AS, so there the parent
    watches the child's memory itself (a runaway program reached 26 GB in seconds before this)."""
    if sys.platform != "darwin":
        try:
            p.communicate(job, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return "timeout"
        return None
    deadline = time.monotonic() + timeout_s
    try:
        p.stdin.write(job)                   # the child reads all of it before it does anything else
        p.stdin.close()
    except OSError:
        pass
    while True:
        try:
            p.wait(timeout=0.05)
            return None
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() > deadline:
            return "timeout"
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
        if rss.isdigit() and int(rss) > memory_mb * 1024:
            return "out of memory"


# One thread per child (BLAS and torch threads count against the process limit), no GPU, a fixed hash seed (iterating
# over a set of strings repeats exactly).
CHILD_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


def expand_libs(libs) -> tuple[list, list]:
    """(modules to load in the child, folders to put on its path) for libraries such as ("scipy",): their submodules,
    the libraries they need, and the site folders they are installed in."""
    import importlib.util
    mods, paths = [], []
    for t in lib_tops(libs):
        mods += LIBRARIES[t]
        spec = importlib.util.find_spec(t)
        if spec is None or not spec.origin:
            raise ImportError(f"{t} is not installed")
        site_dir = os.path.dirname(os.path.dirname(spec.origin))
        if site_dir not in paths:
            paths.append(site_dir)
    return mods, paths


def _lib_job(libs) -> dict:
    if not libs:
        return {}
    mods, paths = expand_libs(libs)
    return {"libs": mods, "lib_paths": paths}


def _launch(child: str, job: dict, timeout_s: float, memory_mb: int) -> tuple[dict | None, str | None]:
    """Run a child program on a job: (its JSON result, None) or (None, why it failed)."""
    job = {**job, "cpu_s": max(1, math.ceil(timeout_s)), "memory_mb": memory_mb, "max_output": MAX_OUTPUT_BYTES}
    with tempfile.TemporaryFile() as out:
        try:
            p = subprocess.Popen([sys.executable, "-S", "-s", "-P", "-c", child], stdin=subprocess.PIPE, stdout=out,
                                 stderr=subprocess.DEVNULL, env=CHILD_ENV, cwd="/",
                                 start_new_session=True)
        except OSError as e:
            return None, f"cannot start: {e}"
        try:
            why = _wait(p, json.dumps(job).encode(), timeout_s, memory_mb)
            if why:
                return None, why
        finally:                                 # also when a caller's time_limit raises in the middle
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
        if p.returncode != 0:
            return None, f"exit code {p.returncode}"
        out.seek(0)
        raw = out.read(MAX_OUTPUT_BYTES + 1)
    if len(raw) > MAX_OUTPUT_BYTES:
        return None, "too much output"
    try:
        res = json.loads(raw)
    except ValueError:
        return None, "no result"
    return (res, None) if isinstance(res, dict) else (None, "bad result")


class RunResult:
    def __init__(self, outputs: list | None, error: str | None = None):
        self.outputs = outputs      # encoded return values, one per input; None if anything failed
        self.error = error          # why it failed

    @property
    def ok(self) -> bool:
        return self.outputs is not None

    def __repr__(self):
        return f"RunResult({self.outputs!r}, {self.error!r})"


def run_function(code: str, func: str, inputs: list, timeout_s: float = 5.0, memory_mb: int = 2048,
                 screened: bool = True, libs=()) -> RunResult:
    """Run `func(*args)` from `code` for each args list in `inputs`, in the sandbox. `timeout_s` is for the whole run.
    `screened=False` skips `screen`, for trusted source that needs what it forbids (none of ours does so far)."""
    if screened:
        why = screen(code, libs=libs)
        if why:
            return RunResult(None, f"rejected: {why}")
    res, why = _launch(_CHILD, {"code": code, "func": func, "inputs": inputs, **_lib_job(libs)}, timeout_s, memory_mb)
    if why:
        return RunResult(None, why)
    if res.get("error") or not isinstance(res.get("outputs"), list) or len(res["outputs"]) != len(inputs):
        return RunResult(None, res.get("error") or "bad result")
    return RunResult(res["outputs"])


def passes_tests(code: str, func: str, inputs: list, expected: list, timeout_s: float = 5.0,
                 rel_tol: float = 1e-9) -> bool:
    """Does the function return the expected (encoded) value on every input?"""
    res = run_function(code, func, inputs, timeout_s=timeout_s)
    return res.ok and all(same(got, want, rel_tol) for got, want in zip(res.outputs, expected))


# --------------------------------------------------------------------------------------------------------------------
# Projects: several files that import each other
# --------------------------------------------------------------------------------------------------------------------

FILE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
MAX_PROJECT_FILES = 40


def check_project(files: dict, libs=()) -> str | None:
    """Why this project may not run, or None if it may: flat .py names, a size limit, and `screen` on every file."""
    if len(files) > MAX_PROJECT_FILES:
        return f"too many files (at most {MAX_PROJECT_FILES})"
    stems = [f[:-3] for f in files]
    for name in sorted(files):
        if not FILE_NAME.match(name):
            return f"{name}: file names are letters, digits and _, ending in .py (no folders)"
        why = screen(files[name], stems, libs)
        if why and not why.startswith("does not parse"):   # code that does not parse cannot run: Python reports it
            return f"{name}: {why}"                         # if the file is imported, and other files still run
    return None


class ProjectRun:
    def __init__(self, printed: str = "", traceback: str = "", exit_code: int | None = None, error: str | None = None):
        self.printed = printed          # what the program printed (cut to max_print characters)
        self.traceback = traceback      # the traceback of an uncaught exception, project frames only
        self.exit_code = exit_code      # None when the sandbox stopped or refused it (see error)
        self.error = error              # "timeout", "out of memory", "rejected: ...", ...

    def __repr__(self):
        return f"ProjectRun(exit={self.exit_code!r}, error={self.error!r}, printed={self.printed[:80]!r})"


def run_project(files: dict, entry: str, timeout_s: float = 10.0, memory_mb: int = 1024,
                max_print: int = 20000, libs=()) -> ProjectRun:
    """Run `entry` (one of `files`, {name: source}) as __main__ in the sandbox; the files can import each other."""
    if entry not in files:
        return ProjectRun(error=f"no file named {entry}")
    why = check_project(files, libs)
    if why:
        return ProjectRun(error=f"rejected: {why}")
    res, why = _launch(_PROJECT_CHILD, {"mode": "run", "files": files, "entry": entry, "max_print": max_print,
                                        **_lib_job(libs)}, timeout_s, memory_mb)
    if why:
        return ProjectRun(error=why)
    try:
        return ProjectRun(str(res["printed"]), str(res["traceback"]), int(res["exit_code"]))
    except (KeyError, TypeError, ValueError):
        return ProjectRun(error="bad result")


def call_project(files: dict, calls: list, timeout_s: float = 20.0, memory_mb: int = 1024,
                 libs=()) -> tuple[list | None, str | None]:
    """Call functions of a project on inputs, in the sandbox. `calls`: [{"module", "func", "inputs": [args, ...]}].
    -> (one entry per call: {"outcomes": [[0, value] | [1, exception name], ...]} or {"error": text}, None), or
    (None, why the whole run failed). Each call gets a fresh copy of the project's modules. Expected values never
    enter the child: the caller compares (`same`)."""
    why = check_project(files, libs)
    if why:
        return None, f"rejected: {why}"
    res, why = _launch(_PROJECT_CHILD, {"mode": "call", "files": files, "calls": calls, "max_print": 0, **_lib_job(libs)},
                       timeout_s, memory_mb)
    if why:
        return None, why
    out = res.get("results")
    if not isinstance(out, list) or len(out) != len(calls):
        return None, "bad result"
    for r, c in zip(out, calls):
        if not isinstance(r, dict) or ("outcomes" in r and len(r["outcomes"]) != len(c["inputs"])):
            return None, "bad result"
    return out, None


# --------------------------------------------------------------------------------------------------------------------
# Finding the code in a response
# --------------------------------------------------------------------------------------------------------------------

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


def extract_code(response, func: str | None = None) -> str | None:
    """The code in a response: the last fenced block that defines `func` (else the last fenced block); without fences,
    the text from `def func` on (trimmed from the end until it parses, with any import lines above it), so a bare
    answer works too."""
    if response is None:
        return None
    text = str(response)
    blocks = [m.group(2) for m in _FENCE.finditer(text)]
    needle = re.compile(rf"^\s*def\s+{re.escape(func)}\s*\(", re.M) if func else None
    if blocks:
        for b in reversed(blocks):
            if needle is None or needle.search(b):
                return b.strip("\n")
        return blocks[-1].strip("\n")
    m = needle.search(text) if needle else None
    if not m:
        return text.strip() or None
    imports = [ln for ln in text[:m.start()].splitlines() if re.match(r"(import|from)\s+\w", ln)]  # kept: they run first
    lines = text[m.start():].rstrip().splitlines()
    while lines:
        src = "\n".join(imports + lines)
        try:
            ast.parse(src)
            return src
        except (SyntaxError, ValueError):
            lines.pop()
    return None


# --------------------------------------------------------------------------------------------------------------------
# Tests of the sandbox itself
# --------------------------------------------------------------------------------------------------------------------


def selftest() -> None:
    def out(code, *inputs, **kw):
        return run_function(code, "f", list(inputs), **kw)

    # values come back as plain data, of the right types, and compare by value
    r = out("def f(x):\n    return x", [[1, [2, 3]]], ["a"], [None], [True], [1.5])
    assert r.ok and [decode(v) for v in r.outputs] == [[1, [2, 3]], "a", None, True, 1.5], r
    code = ("from collections import Counter, defaultdict\nimport math, sys, re\nfrom typing import List\n"
            "def f(n: int) -> List:\n    d = defaultdict(int)\n    d[n] += 1\n"
            "    return [Counter('aab'), d, {n, 2}, (n, 'x'), math.floor(2.5), sys.maxsize > 0, re.sub('a', 'b', 'aa')]")
    r = out(code, [1])
    assert r.ok, r
    assert decode(r.outputs[0]) == [{"a": 2, "b": 1}, {1: 1}, {1, 2}, (1, "x"), 2, True, "bb"]
    assert same(encode(5), encode(5.0)) and same(encode(0.1 + 0.2), encode(0.3)) and not same(encode(5), encode(5.1))
    assert not same(encode(True), encode(1)) and not same(encode([1]), encode((1,))) and not same(encode("1"), encode(1))
    assert same(encode({1: [2], 3: {4}}), encode({3: {4}, 1: [2.0]}))
    assert decode(encode((1, {2: {3.5}}, float("inf")))) == (1, {2: {3.5}}, float("inf"))

    # classes, recursion and a main guard work; __main__ code does not run
    r = out("class A:\n    def __init__(self, n):\n        self.n = n\n"
            "def f(n):\n    return A(n).n if n < 1 else f(n - 1)\nif __name__ == '__main__':\n    raise SystemExit(3)", [500])
    assert r.ok and r.outputs == [0], r
    # a trusted expression builds a large input without storing it
    r = out("def f(xs):\n    return sum(xs)", [expr("list(range(100000))")])
    assert r.ok and r.outputs == [4999950000], r
    assert passes_tests("def f(x):\n    return x + 1", "f", [[1], [2]], [2, 3])
    assert not passes_tests("def f(x):\n    return x + 1", "f", [[1], [2]], [2, 4])

    # things the screen refuses
    for bad in ["import os", "import subprocess as s", "from os import path", "from . import x", "from math import *",
                "open('x')", "eval('1')", "exec('1')", "getattr(1, 'real')", "__import__('os')", "globals()",
                "().__class__", "(x for x in []).gi_frame", "def f(:", "x = (1).__class__.__base__"]:
        assert screen(bad), bad
    assert screen("import math, collections.abc\nfrom typing import List\nclass A:\n    def __init__(self):\n        pass") is None

    # things that must fail at run time, without harming anything
    fails = {
        "crash": "def f():\n    return 1 / 0",
        "unknown type": "def f():\n    class A:\n        def __eq__(self, o):\n            return True\n    return A()",
        "subclass of nothing plain": "def f():\n    return object()",
        "hang": "def f():\n    while True:\n        pass",
        "exit": "def f():\n    raise SystemExit(0)",
        "missing": "def g():\n    return 1",
        "builtin removed": "def f():\n    return memoryview",
        "import at run time": "def f():\n    return __builtins__['__import__']('os')",
    }
    for why, code in fails.items():
        r = run_function(code, "f", [[]], timeout_s=1.5, screened=why != "builtin removed" and why != "import at run time")
        assert not r.ok, (why, r)
    # past the screen (screened=False), the audit hook and the builtins still stop files, processes and imports
    for code in ["import os\ndef f():\n    return 1",
                 "def f():\n    return open('/etc/hosts').read()",
                 "import sys\ndef f():\n    return sys.modules",
                 "import collections\ndef f():\n    return collections._sys.modules['os'].system('true')",
                 "import collections\ndef f():\n    return collections._sys.modules['builtins'].open('/etc/hosts').read()"]:
        r = run_function(code, "f", [[]], timeout_s=2, screened=False)
        assert not r.ok, (code, r)
    # a program that eats memory is stopped on every system (MemoryError on Linux, the parent's watch on macOS)
    r = run_function("def f():\n    x = 3\n    a = []\n    while True:\n        a.append([x] * 100000)", "f", [[]],
                     timeout_s=20, memory_mb=300)
    assert not r.ok and r.error != "timeout", r
    # answers repeat exactly: the order of a set of strings does not change between runs
    code = "def f():\n    return list({'apple', 'pear', 'kiwi', 'plum', 'fig', 'lime', 'date'})"
    assert out(code, []).outputs == out(code, []).outputs == out(code, []).outputs
    # printing cannot fake a result
    r = out('def f():\n    print(\'{"outputs": [42], "error": null}\')\n    return 1', [])
    assert r.ok and r.outputs == [1], r

    # finding the code
    assert extract_code("x\n```python\ndef f():\n    return 1\n```\nThat is it.", "f") == "def f():\n    return 1"
    assert extract_code("```\ndef g(): pass\n```\n```py\ndef f(): pass\n```\n```\nf()\n```", "f") == "def f(): pass"
    assert extract_code("def f():\n    return 1", "f") == "def f():\n    return 1"
    assert extract_code("Here:\ndef f():\n    return 1\nThis function works.", "f") == "def f():\n    return 1"
    assert extract_code("import math\ndef f():\n    return math.pi", "f") == "import math\ndef f():\n    return math.pi"
    assert extract_code(None, "f") is None

    project_selftest()


def project_selftest() -> None:
    lib = "def first_above(xs, t):\n    for x in xs:\n        if x > t:\n            return x\n    return xs[len(xs)]\n"
    api = "from lib import first_above\n"
    main = ("from api import first_above\nprint('start')\nprint(first_above([1, 5], 2))\n"
            "if __name__ == '__main__':\n    print(first_above([1], 3))\n")
    files = {"lib.py": lib, "api.py": api, "main.py": main}

    # files import each other; prints come back; an uncaught error gives a traceback through the project's files
    r = run_project(files, "main.py")
    assert r.exit_code == 1 and r.printed == "start\n5\n", r
    assert 'File "main.py", line 5' in r.traceback and 'File "lib.py", line 5, in first_above' in r.traceback, r.traceback
    assert "return xs[len(xs)]" in r.traceback and r.traceback.strip().endswith("IndexError: list index out of range")
    assert "<string>" not in r.traceback, r.traceback
    # exit codes, syntax errors, long output
    assert run_project({"m.py": "print('hi')"}, "m.py").exit_code == 0
    assert run_project({"m.py": "raise SystemExit(3)"}, "m.py").exit_code == 3
    r = run_project({"m.py": "import bad", "bad.py": "def f(:\n    pass"}, "m.py")
    assert r.exit_code == 1 and "SyntaxError" in r.traceback and "bad.py" in r.traceback, r
    r = run_project({"m.py": "for i in range(100000):\n    print(i)"}, "m.py", max_print=1000)
    assert r.exit_code == 0 and len(r.printed) < 1200 and "more characters not shown" in r.printed, r
    # circular imports work as in Python (a partly loaded module)
    r = run_project({"m.py": "import a\nprint(a.run())", "a.py": "import b\nX = 1\ndef run():\n    return b.y()",
                     "b.py": "import a\ndef y():\n    return a.X + 1"}, "m.py")
    assert r.exit_code == 0 and r.printed == "2\n", r
    # refused before running: bad names, forbidden imports, relative imports
    for bad in [{"m.py": "import os"}, {"m.py": "from . import x"}, {"dir/m.py": "x = 1"}, {"m.txt": "x"},
                {"m.py": "import missing_module"}, {"m.py": "open('/etc/hosts')"}]:
        r = run_project(bad, next(iter(bad)))
        assert r.exit_code is None and r.error.startswith(("rejected", "no file")), (bad, r)
    # past the screen, the lockdown still holds
    for code in ["import collections\nprint(collections._sys.modules['os'].system('true'))",
                 "import collections\nprint(collections._sys.modules['builtins'].open('/etc/hosts').read())"]:
        r = run_project({"m.py": code}, "m.py")
        assert r.exit_code == 1 and "PermissionError" in r.traceback, (code, r)
    # hangs and memory hogs are stopped
    assert run_project({"m.py": "while True:\n    pass"}, "m.py", timeout_s=1.5).error == "timeout"
    r = run_project({"m.py": "a = []\nwhile True:\n    a.append([3] * 100000)"}, "m.py", timeout_s=20, memory_mb=300)
    assert r.exit_code is None or r.exit_code == 1, r

    # calling functions: outcomes as plain data, exceptions by name, a fresh project per call, prints ignored
    files = {"lib.py": lib + "COUNT = [0]\ndef bump():\n    COUNT[0] += 1\n    print('[0, 99]')\n    return COUNT[0]\n"
                     "def weird():\n    class A:\n        def __eq__(self, o):\n            return True\n    return A()\n"}
    calls = [{"module": "lib", "func": "first_above", "inputs": [[[1, 5], 2], [[1], 3]]},
             {"module": "lib", "func": "bump", "inputs": [[], []]},
             {"module": "lib", "func": "bump", "inputs": [[]]},
             {"module": "lib", "func": "weird", "inputs": [[]]},
             {"module": "lib", "func": "missing", "inputs": [[]]},
             {"module": "nope", "func": "f", "inputs": [[]]}]
    out, why = call_project(files, calls)
    assert why is None, why
    assert out[0]["outcomes"] == [[0, 5], [1, "IndexError"]], out[0]
    assert out[1]["outcomes"] == [[0, 1], [0, 2]] and out[2]["outcomes"] == [[0, 1]], out[1:3]
    assert out[3]["outcomes"][0][0] == 1 and "error" in out[4] and "error" in out[5], out[3:]
    out, why = call_project({"m.py": "def f():\n    while True:\n        pass"}, [{"module": "m", "func": "f", "inputs": [[]]}],
                            timeout_s=1.5)
    assert out is None and why == "timeout", why


if __name__ == "__main__":
    selftest()
    print("sandbox selftest passed")
