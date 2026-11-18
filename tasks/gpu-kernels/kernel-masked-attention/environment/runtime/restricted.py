"""Restricted numerical Python edit surface.

This language rejects unsupported syntax before execution. File and process
APIs, timers, reflection, and mutable module objects are unavailable.
"""
import ast,builtins,math,types
from pathlib import Path
import torch
from torch.nn import functional as F

ATTRIBUTES=set('''shape dtype device is_cuda ndim T numel size stride dim float double half bfloat16 to type_as
detach clone contiguous view reshape flatten transpose permute unsqueeze squeeze expand expand_as repeat repeat_interleave
chunk split unbind sum mean max min amax amin square sqrt rsqrt exp log abs clamp clamp_min clamp_max softmax log_softmax
masked_fill where gather scatter index_select nonzero any all bool long int item new_empty new_zeros new_ones
register_post_accumulate_grad_hook register_hook remove wait requires_grad_ requires_grad grad zero_ fill_ copy_ add_ sub_ mul_ div_ lerp_ addcmul_ addcdiv_ norm
sort argsort cumsum searchsorted index_add_ get values items keys setdefault append extend pop backward apply save_for_backward saved_tensors needs_input_grad
wte wpe blocks head ln parameters named_parameters state_dict get_submodule train eval weight bias
base a b in_features out_features forward'''.split())
TORCH_NAMES='searchsorted Tensor tensor empty empty_like zeros zeros_like ones ones_like full full_like arange linspace eye stack cat concat matmul mm bmm einsum sigmoid softmax log_softmax exp log sqrt rsqrt where maximum minimum clamp tril triu isfinite finfo float32 float64 float16 bfloat16 int32 int64 long bool compile no_grad'.split()
FUNCTIONAL_NAMES='binary_cross_entropy_with_logits kl_div scaled_dot_product_attention linear gelu silu softmax log_softmax cross_entropy logsigmoid mse_loss layer_norm'.split()
TRITON_NAMES='jit cdiv next_power_of_2 autotune Config heuristics'.split()
TL_NAMES='constexpr arange program_id num_programs load store dot full zeros exp exp2 log log2 sqrt rsqrt sum max min maximum minimum where cast float16 bfloat16 float32 float64 int8 int16 int32 int64 uint32 uint64 multiple_of max_contiguous assume static_range range trans broadcast_to advance make_block_ptr inline_asm_elementwise swizzle2d'.split()
ATTRIBUTES.update(TORCH_NAMES+FUNCTIONAL_NAMES+TRITON_NAMES+TL_NAMES)
ATTRIBUTES.update('argmax argmin topk round log1p tanh sigmoid softplus index_copy clone detach tolist count join strip split startswith endswith replace indices cos sin index_add view_as topk argmax argmin pow floor remainder div median std var logsumexp reciprocal normal_ uniform_ int8 uint8 numel element_size storage_offset'.split())
TORCH_NAMES += 'argmax argmin topk rand randn randint multinomial nonzero int8 uint8'.split()
ATTRIBUTES.update(TORCH_NAMES)
ATTRIBUTES.update(['distributed','all_reduce','get_world_size','get_rank','barrier','nn','functional','autograd','Function','language','utils','checkpoint'])
ATTRIBUTES.update(['cos','sin','pi','ceil','floor'])


class Facade:
    __slots__=('_values',)
    def __init__(self,values):object.__setattr__(self,'_values',values)
    def __getattr__(self,name):
        values=object.__getattribute__(self,'_values')
        if name not in values:raise AttributeError('API not available: '+name)
        return values[name]
    def __setattr__(self,name,value):raise TypeError('runtime namespaces are immutable')
    def __copy__(self):return self
    def __deepcopy__(self,memo):return self


def validate(source,required):
    if len(source.encode())>256*1024:raise ValueError('source exceeds 256 KiB')
    tree=ast.parse(source)
    banned=(ast.Global,ast.Nonlocal,ast.Try,ast.TryStar,ast.With,ast.AsyncWith,ast.AsyncFunctionDef,ast.Await,ast.Yield,ast.YieldFrom,ast.Delete)
    for n in ast.walk(tree):
        if isinstance(n,banned):raise ValueError('unsupported numerical-language syntax: '+type(n).__name__)
        if isinstance(n,ast.Name) and (n.id.startswith('__') or n.id in {'getattr','setattr','delattr','eval','exec','compile','globals','locals','vars','dir','type','object','super','open','input','breakpoint'}):raise ValueError('reflection names are unavailable')
        if isinstance(n,ast.Attribute) and n.attr not in ATTRIBUTES:raise ValueError('attribute outside numerical API: '+n.attr)
        if isinstance(n,ast.FunctionDef) and n.name.startswith('__'):raise ValueError('special methods are unavailable')
        if isinstance(n,ast.ClassDef):
            if len(n.bases)!=1 or ast.unparse(n.bases[0])!='torch.autograd.Function':raise ValueError('only autograd.Function classes are supported')
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.For,ast.AsyncFor,ast.comprehension,ast.NamedExpr)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(v,ast.Attribute) for t in targets for v in ast.walk(t)):raise ValueError('attribute assignment is unavailable')
        if isinstance(n,ast.Import):
            if any(x.name not in ('torch','math','triton','triton.language') for x in n.names):raise ValueError('import outside numerical API')
        if isinstance(n,ast.ImportFrom):
            allowed={'torch.nn':{'functional'},'kernel_ops':{'reference','attention'},'basic_gemm':{'matmul'},'packing':{'padded'},'torch.utils.checkpoint':{'checkpoint'}}
            if n.level or n.module not in allowed or any(x.name not in allowed[n.module] for x in n.names):raise ValueError('import outside numerical API')
    names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    if not set(required)<=names:raise ValueError('missing required entrypoint')
    return tree


def load(path,required):
    path=Path(path);source=path.read_text();tree=validate(source,required)
    from kernel_ops import reference,attention
    from packing import padded
    from basic_gemm import matmul
    from torch.utils.checkpoint import checkpoint
    import torch.distributed as dist
    distributed=Facade({n:getattr(dist,n) for n in ('all_reduce','get_world_size','get_rank','barrier')})
    functional=Facade({n:getattr(F,n) for n in FUNCTIONAL_NAMES})
    torch_api={n:getattr(torch,n) for n in TORCH_NAMES}
    torch_api.update(distributed=distributed,nn=Facade({'functional':functional}),autograd=Facade({'Function':torch.autograd.Function}),utils=Facade({'checkpoint':Facade({'checkpoint':checkpoint})}))
    modules={'torch':Facade(torch_api),'torch.nn':Facade({'functional':functional}),
      'math':Facade({n:getattr(math,n) for n in ['sqrt','exp','log','log2','cos','sin','pi','ceil','floor']}),
      'kernel_ops':Facade({'reference':reference,'attention':attention}),'packing':Facade({'padded':padded}),
      'basic_gemm':Facade({'matmul':matmul}),'torch.utils.checkpoint':Facade({'checkpoint':checkpoint})}
    def importer(name,globals=None,locals=None,fromlist=(),level=0):
        if name in ('triton','triton.language'):
            import triton,triton.language as tl
            language=tl
            triton_api=Facade({**{n:getattr(triton,n) for n in TRITON_NAMES},'language':language})
            return language if name=='triton.language' and fromlist else triton_api
        if name not in modules:raise ImportError('module unavailable: '+name)
        return modules[name]
    safe={n:getattr(builtins,n) for n in ['range','len','min','max','sum','zip','enumerate','str','sorted','next','int','float','bool','abs','round','tuple','list','dict','isinstance','ValueError','RuntimeError','staticmethod']}
    safe['__import__']=importer
    # __build_class__ is needed only for the narrowly validated autograd class.
    safe['__build_class__']=builtins.__build_class__
    scope={'__builtins__':safe,'__name__':'numerical_submission','__file__':str(path)}
    exec(compile(tree,str(path),'exec'),scope,scope)
    return types.SimpleNamespace(**{n:scope[n] for n in required})
