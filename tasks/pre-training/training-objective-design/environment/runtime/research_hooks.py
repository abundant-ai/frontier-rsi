"""Stateless numerical hook language for the new research interfaces."""
import ast
from pathlib import Path
from restricted import load as numerical_load

def load(path,required,allow_attention=False):
 tree=ast.parse(Path(path).read_text())
 for node in tree.body:
  if not isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):raise ValueError('research source contains imports and functions only')
 for node in ast.walk(tree):
  if isinstance(node,ast.Import) and any(alias.name not in ('torch','math') for alias in node.names):raise ValueError('research imports are limited to torch and math')
  if isinstance(node,ast.ImportFrom) and (node.module!='torch.nn' or any(alias.name!='functional' for alias in node.names)):raise ValueError('research imports are limited to tensor arithmetic')
  if isinstance(node,ast.FunctionDef):
   for default in (*node.args.defaults,*node.args.kw_defaults):
    if default is not None and not isinstance(default,ast.Constant):raise ValueError('mutable/computed function defaults are unavailable')
  if isinstance(node,ast.Attribute) and node.attr in ('register_hook','register_post_accumulate_grad_hook','requires_grad_','backward','compile'):raise ValueError('hooks cannot retain tensors or control the trusted backward pass')
  if isinstance(node,ast.Attribute) and node.attr=='scaled_dot_product_attention' and not allow_attention:raise ValueError('fused attention is unavailable for this task')
 return numerical_load(path,required)
