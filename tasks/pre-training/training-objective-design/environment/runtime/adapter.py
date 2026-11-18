"""Trusted bounded-parameter adaptation; candidates never receive a model."""
import ast
import math
from pathlib import Path
import torch
from torch.nn.utils import parametrize
from research_hooks import load as numerical_load

CAP = 15_000_000


def load(path, required=('configure', 'delta'), allow_attention=False):
    tree = ast.parse(Path(path).read_text())
    # In addition to the stateless language, prohibit mutation and autograd
    # control. Only root-created leaf tensors participate in optimization.
    forbidden = {'grad', 'requires_grad', 'detach', 'backward', 'apply',
                 'register_hook', 'register_post_accumulate_grad_hook',
                 'parameters', 'named_parameters', 'state_dict'}
    for node in ast.walk(tree):
        if isinstance(node,ast.keyword) and node.arg in (None,'requires_grad'):
            raise ValueError('only the trusted runner may create trainable leaf tensors; keyword unpacking is unavailable')
        if isinstance(node,(ast.ClassDef,ast.NamedExpr,ast.Lambda)):
            raise ValueError('bounded hooks cannot retain hidden callable state')
        if isinstance(node,ast.FunctionDef):
            if node.decorator_list or node not in tree.body:
                raise ValueError('bounded hooks use undecorated top-level functions only')
            annotations=[node.returns,*[a.annotation for a in (*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs)]]
            if node.args.vararg:annotations.append(node.args.vararg.annotation)
            if node.args.kwarg:annotations.append(node.args.kwarg.annotation)
            if any(a is not None and not isinstance(a,(ast.Name,ast.Attribute,ast.Constant)) for a in annotations):
                raise ValueError('computed annotations cannot initialize hidden state')
        if isinstance(node, ast.Attribute) and (node.attr in forbidden or node.attr.endswith('_')):
            raise ValueError('adapter hooks cannot mutate tensors or control training')
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(item, (ast.Subscript, ast.Attribute)) for target in targets for item in ast.walk(target)):
                raise ValueError('adapter hooks cannot mutate inputs')
    return numerical_load(path, required, allow_attention=allow_attention)


class Update(torch.nn.Module):
    def __init__(self, name, hook, factors):
        super().__init__()
        self.name, self.hook = name, hook
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, original):
        update = self.hook(self.name, tuple(self.factors))
        if not isinstance(update, torch.Tensor) or update.shape != original.shape:
            raise ValueError('delta must return exactly the selected parameter shape')
        if update.dtype != original.dtype or update.device != original.device:
            raise ValueError('delta must preserve parameter dtype and device')
        return original + update


def prepare(model, candidate, cap=CAP):
    model.requires_grad_(False)
    # Names and integer shapes only: no module, mutable backbone, data, files,
    # optimizer, evaluator, or training graph reaches candidate configuration.
    eligible={};aliases={}
    for prefix,module in model.named_modules():
        for key,parameter in module.named_parameters(recurse=False,remove_duplicate=False):
            if not parameter.is_floating_point():continue
            name=(prefix+'.' if prefix else '')+key
            eligible[name]=(module,key,parameter)
            aliases.setdefault(id(parameter),[]).append((module,key))
    shapes = {name: tuple(parameter.shape) for name, (_,_,parameter) in eligible.items()}
    config = candidate.configure(shapes)
    if type(config) is not dict or set(config) != {'updates', 'learning_rate', 'weight_decay'}:
        raise ValueError('configure returns updates, learning_rate, weight_decay')
    lr, wd = config['learning_rate'], config['weight_decay']
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= .1:
        raise ValueError('invalid learning rate')
    if type(wd) not in (int, float) or not math.isfinite(wd) or not 0 <= wd <= 1:
        raise ValueError('invalid weight decay')
    updates = config['updates']
    if type(updates) is not dict or not updates or len(updates) > len(eligible):
        raise ValueError('updates must select existing model parameters')
    total = 0;selected=set()
    # Validate the entire declaration and count BEFORE allocating any factors.
    for name, specs in updates.items():
        if name not in eligible or type(specs) not in (tuple, list) or not 1 <= len(specs) <= 8:
            raise ValueError('invalid update declaration')
        identity=id(eligible[name][2])
        if identity in selected:raise ValueError('select only one name for a tied model parameter')
        selected.add(identity)
        for spec in specs:
            if type(spec) is not dict or set(spec) != {'shape', 'std'}:
                raise ValueError('factor declares shape and initialization std')
            shape, std = spec['shape'], spec['std']
            if type(shape) not in (tuple, list) or not 1 <= len(shape) <= 4 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError('invalid factor shape')
            if type(std) not in (float, int) or not math.isfinite(std) or not 0 <= std <= 1:
                raise ValueError('invalid initialization std')
            total += math.prod(shape)
            if total > cap:
                raise ValueError('trainable-parameter cap')
    trainable = []
    for name, specs in updates.items():
        module,key,original = eligible[name]
        factors = [torch.nn.Parameter(torch.randn(tuple(spec['shape']), device=original.device,
                                                  dtype=original.dtype) * spec['std']) for spec in specs]
        update=Update(name,candidate.delta,factors)
        for tied_module,tied_key in aliases[id(original)]:
            parametrize.register_parametrization(tied_module,tied_key,update)
        trainable.extend(factors)
    assert sum(t.numel() for t in trainable) == total <= cap
    return tuple(trainable), {'lr': float(lr), 'weight_decay': float(wd), 'trainable_parameters': total}


def export(model):
    # Materialize ordinary original-architecture weights, with no submitted
    # exporter and no candidate-controlled state_dict/parameters methods.
    state = {};materialized={}
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module):
            for key,entries in module.parametrizations.items():
                identity=id(entries.original)
                if identity not in materialized:
                    materialized[identity]=getattr(module,key).detach().cpu().contiguous().clone()
                state[(name+'.' if name else '')+key]=materialized[identity].clone()
    for name, tensor in model.state_dict().items():
        if 'parametrizations.' not in name:
            state[name] = tensor.detach().cpu().contiguous().clone()
    return state
