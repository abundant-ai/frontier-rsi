"""Candidate shape checks and module loading. Not a security sandbox."""
import ast,importlib.util
from pathlib import Path


def validate(path,allowed):
 source=Path(path).read_text()
 if len(source.encode())>256*1024:raise ValueError('candidate.py exceeds 256 KiB')
 tree=ast.parse(source)
 # Require the declared entrypoints and source-size limit.
 functions={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 if not set(allowed)<=functions:raise ValueError(f'Required entrypoints: {allowed}')
 return source


def load(path,allowed):
 validate(path,allowed)
 spec=importlib.util.spec_from_file_location('submission',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
