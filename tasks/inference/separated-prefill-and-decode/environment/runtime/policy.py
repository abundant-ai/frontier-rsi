"""Small pure-Python policy language: no model, timer, filesystem or reflection.

The trusted runtime passes only immutable primitive summaries. All decisions are
validated before they affect the model or persistent cache.
"""
import ast
import builtins
import hashlib
import os
import stat
from pathlib import Path


class CandidateError(ValueError):
    pass


def load(path, entrypoint="plan", *, source=None):
    if source is None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as exc:
            raise CandidateError("candidate must be a readable regular source file") from exc
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                raise CandidateError("candidate must be a regular file of at most 64 KiB")
            encoded = handle.read(65537)
    else:
        encoded = source
    if len(encoded) > 65536:
        raise CandidateError("candidate exceeds 64 KiB")
    try:
        source = encoded.decode('utf-8')
    except UnicodeError as exc:
        raise CandidateError("candidate source must be UTF8") from exc
    # Parse and compile failures belong to the submitted source.
    try:
        tree = ast.parse(source)
    except Exception as exc:
        raise CandidateError("invalid candidate syntax") from exc
    if not tree.body or any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise CandidateError("only function definitions are permitted at module scope")
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Try,
                 ast.TryStar, ast.With, ast.AsyncWith, ast.AsyncFunctionDef,
                 ast.Await, ast.Yield, ast.YieldFrom, ast.ClassDef, ast.NamedExpr)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise CandidateError("unsupported syntax: " + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CandidateError("reflection is unavailable")
        if isinstance(node, ast.Attribute) and (not isinstance(node.ctx, ast.Load) or node.attr not in ("append", "pop", "get", "items", "keys", "values", "count", "index")):
            raise CandidateError("unsupported attribute: " + node.attr)
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            args += tuple(a for a in (node.args.vararg, node.args.kwarg) if a is not None)
            if isinstance(node, ast.FunctionDef) and (node.name.startswith("__") or node.decorator_list or node.returns):
                raise CandidateError("decorators, annotations and special names are unavailable")
            if any(a.annotation for a in args):
                raise CandidateError("annotations are unavailable")
            if node.args.defaults or any(v is not None for v in node.args.kw_defaults):
                raise CandidateError("function defaults are unavailable")
    if entrypoint not in {node.name for node in tree.body}:
        raise CandidateError("missing " + entrypoint)
    names = "abs all any bool enumerate float int len list dict max min range round sorted sum tuple zip set".split()
    scope = {"__builtins__": {name: getattr(builtins, name) for name in names}}
    try:
        code = compile(tree, str(path), "exec")
        exec(code, scope, scope)
    except Exception as exc:
        raise CandidateError("invalid candidate source") from exc
    function = scope[entrypoint]
    function.source_sha256 = hashlib.sha256(encoded).hexdigest()
    function._source_bytes = encoded
    function._source_path = str(path)
    function._entrypoint = entrypoint
    return function


def fresh(function):
    return load(function._source_path, function._entrypoint, source=function._source_bytes)


def call(function, *args):
    from execution_guard import deadline
    with deadline(.1):
        try:
            return function(*args)
        except Exception as exc:
            raise CandidateError(f"policy raised {type(exc).__name__}: {exc}") from exc
