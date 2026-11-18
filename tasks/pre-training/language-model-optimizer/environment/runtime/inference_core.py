"""Frozen-model services. Candidate hooks have numerical capabilities only."""
import contextlib,json,math,time,types
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen2Config,Qwen2ForCausalLM
from transformers.cache_utils import Cache,DynamicLayer,DynamicCache
from research_model import storage_bytes
from research_hooks import load
from train import answer

HOOKS=['execute_layer','propose','encode','decode','retain','compress','schedule','plan','calibrate']

def sync(device):
 if str(device).startswith('cuda'):torch.cuda.synchronize()


def frozen(assets,device,smoke=False):
 if smoke:
  torch.manual_seed(19)
  model=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=65536)).to(device)
  return model.eval().requires_grad_(False),None
 tok=AutoTokenizer.from_pretrained(Path(assets)/'model',local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(Path(assets)/'model',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
 return model,tok


def merge_encoded(left,right,new_tokens):
 # Generic tensor representations may have non-token metadata. Only coalesce
 # an explicit tuple/list of tensors whose token dimension is preserved.
 if not isinstance(left,(tuple,list)) or not isinstance(right,type(left)) or len(left)!=len(right):return None
 for a,b in zip(left,right):
  if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or a.ndim<3 or a.ndim!=b.ndim or b.shape[2]!=new_tokens:return None
  if a.device!=b.device or a.dtype!=b.dtype or a.shape[:2]+a.shape[3:]!=b.shape[:2]+b.shape[3:]:return None
  if a.shape[2]!=left[0].shape[2]:return None
 return type(left)(torch.cat((a,b),dim=2) for a,b in zip(left,right))


class StateLayer(DynamicLayer):
 def __init__(self,hooks,mode,capacity):
  super().__init__();self.hooks=hooks;self.mode=mode;self.capacity=capacity;self.payload=None;self.n=0
 def update(self,k,v,cache_kwargs=None):
  if self.payload is not None:
   if self.mode==34:
    oldk=torch.cat([self.hooks.decode(chunk).to(k.dtype) for chunk in self.payload[0]],-2)
    oldv=torch.cat([self.hooks.decode(chunk).to(v.dtype) for chunk in self.payload[1]],-2)
   else:oldk,oldv=self.payload
   k=torch.cat((oldk,k),-2);v=torch.cat((oldv,v),-2)
  self.pending=(k,v);return k,v
 def finalize(self,position):
  k,v=self.pending
  if self.mode==34:
   # Re-encoding the entire cache after every step repeatedly rounds old
   # four-bit keys/values. Preserve the existing encoded chunks and encode
   # only the newly appended positions.
   fresh=(k[...,self.n:,:],v[...,self.n:,:])
   chunks=([[],[]] if self.payload is None else self.payload)
   for source,items in zip(fresh,chunks):
    payload=self.hooks.encode(source.clone())
    rebuilt=self.hooks.decode(payload)
    if rebuilt.shape!=source.shape or not torch.isfinite(rebuilt).all():raise ValueError('cache representation decode shape/finiteness')
    merged=merge_encoded(items[-1],payload,source.shape[2]) if items else None
    if merged is not None:items[-1]=merged
    else:items.append(payload)
   self.n=k.shape[-2];self.payload=chunks
   del self.pending
   return
  if self.mode in (32,55):
   indices=self.hooks.retain(k.clone(),v.clone(),self.capacity,position)
   if indices.dtype!=torch.int64 or indices.ndim!=1 or not len(indices) or len(indices)>self.capacity or (indices<0).any() or (indices>=k.shape[-2]).any() or len(torch.unique(indices))!=len(indices):raise ValueError('invalid retained positions')
   indices=indices.sort().values;k=k.index_select(-2,indices).contiguous();v=v.index_select(-2,indices).contiguous()
  self.n=k.shape[-2]
  self.payload=(k,v)
  del self.pending
 def get_seq_length(self):return self.n
 def get_mask_sizes(self,cache_position):return self.n+len(cache_position),0


class StateCache(Cache):
 def __init__(self,model,hooks,mode,batch):
  # Split the per-sequence cap across every layer, both key and value.
  per_token=2*model.config.num_hidden_layers*model.config.num_key_value_heads*(model.config.hidden_size//model.config.num_attention_heads)*2
  capacity=max(1,(64*1024**2)//per_token)
  super().__init__(layers=[StateLayer(hooks,mode,capacity) for _ in range(model.config.num_hidden_layers)])
  self.batch=batch
  self.benchmark_checks_on_device=False
 def finalize(self,position):
  if self.layers[0].mode==32:
   plans=[]
   for layer in self.layers:
    k,v=layer.pending
    indices=layer.hooks.retain(k.clone(),v.clone(),layer.capacity,position)
    if indices.dtype!=torch.int64 or indices.ndim!=1 or not len(indices) or len(indices)>layer.capacity:raise ValueError('invalid retained positions')
    indices=indices.sort().values
    plans.append((layer,k,v,indices))
   # Combine identical-length layer plans into a single GPU validity check.
   # Retain hooks may return different lengths; keep the general fallback.
   if len({len(indices) for _,_,_,indices in plans})==1 and len({k.shape for _,k,_,_ in plans})==1:
    chosen=torch.stack([indices for _,_,_,indices in plans]).sort(-1).values
    upper=plans[0][1].shape[-2]
    bad=((chosen<0)|(chosen>=upper)).any() | (chosen[:,1:]==chosen[:,:-1]).any()
    if self.benchmark_checks_on_device:torch._assert_async(~bad,'invalid retained positions')
    elif bad:raise ValueError('invalid retained positions')
    keys=torch.stack([k for _,k,_,_ in plans]);values=torch.stack([v for _,_,v,_ in plans])
    index=chosen[:,None,None,:,None].expand(keys.shape[0],keys.shape[1],keys.shape[2],chosen.shape[1],keys.shape[-1])
    keys=keys.gather(-2,index);values=values.gather(-2,index)
    for i,(layer,k,v,indices) in enumerate(plans):
     layer.payload=(keys[i],values[i]);layer.n=chosen.shape[1];del layer.pending
   else:
    invalid=[(indices<0).any() | (indices>=k.shape[-2]).any() | (indices[1:]==indices[:-1]).any() for _,k,_,indices in plans]
    bad=torch.stack(invalid).any()
    if self.benchmark_checks_on_device:torch._assert_async(~bad,'invalid retained positions')
    elif bad:raise ValueError('invalid retained positions')
    for layer,k,v,indices in plans:
     layer.payload=(k.index_select(-2,indices).contiguous(),v.index_select(-2,indices).contiguous())
     layer.n=len(indices);del layer.pending
  else:
   for layer in self.layers:layer.finalize(position)
 def bytes(self):return storage_bytes([layer.payload for layer in self.layers])


def cached_call(model,tokens,cache,position):
 n=cache.layers[0].n;t=tokens.shape[1];b=tokens.shape[0]
 # A single new query can attend to every retained key and itself. Avoid
 # allocating a fresh additive attention mask on every decode step.
 mask=None
 if t>1:
  mask=torch.zeros(t,n+t,device=tokens.device,dtype=model.dtype)
  mask[:,n:]=torch.where(torch.ones(t,t,device=tokens.device,dtype=torch.bool).tril(),0.,float('-inf')).to(model.dtype)
  mask=mask[None,None].expand(b,1,-1,-1)
 positions=torch.arange(position,position+t,device=tokens.device)
 result=model(tokens,past_key_values=cache,attention_mask=mask,position_ids=positions[None,:].expand(b,-1),cache_position=positions,use_cache=True,logits_to_keep=1)
 cache.finalize(position+t);return result.logits


def quantize_model(model,hooks,calibration,prepared=None):
 # Candidate representation is tensor-only. Every payload and retained parameter
 # contributes to the payload cap; original dense matrices are removed.
 from tensor_payload import read,write
 cached=read(prepared,model.device) if prepared and (Path(prepared)/'layout.json').exists() else None
 payloads=[];moments={};handles=[];representations={}
 def collect(module,args):
  x=args[0].detach().float().reshape(-1,args[0].shape[-1])
  if module not in moments:moments[module]={'covariance':torch.zeros(x.shape[-1],x.shape[-1],device=x.device),'count':0}
  moments[module]['covariance'].add_(x.transpose(0,1)@x);moments[module]['count']+=len(x)
 for module in model.modules():
  if cached is None and isinstance(module,nn.Linear):handles.append(module.register_forward_pre_hook(collect))
 with torch.no_grad():
  if cached is None:
   for sequence in calibration:model(sequence[None,:],use_cache=False,logits_to_keep=1)
 for handle in handles:handle.remove()
 for module in list(model.modules()):
  if isinstance(module,(nn.Linear,nn.Embedding)):
   parameter=module.weight;weight=parameter.detach();shape=weight.shape;dtype=weight.dtype
   if parameter in representations:
    payload=representations[parameter]
   elif cached is None:
    stats=moments.pop(module,None)
    covariance=stats['covariance']/max(1,stats['count']) if stats else torch.eye(weight.shape[-1],device=weight.device)
    payload=hooks.calibrate(weight.clone(),covariance)
   else:payload=cached[len(payloads)]
   representations[parameter]=payload
   rebuilt=hooks.decode(payload)
   if rebuilt.shape!=shape or not torch.isfinite(rebuilt).all():raise ValueError('invalid weight representation')
   payloads.append(payload);module.weight=nn.Parameter(torch.empty(0,device=weight.device,dtype=dtype),requires_grad=False)
   if isinstance(module,nn.Linear):
    def forward(self,x,payload=payload,dtype=dtype):return nn.functional.linear(x,hooks.decode(payload).to(dtype),self.bias)
   else:
    def forward(self,x,payload=payload,dtype=dtype):return nn.functional.embedding(x,hooks.decode(payload).to(dtype))
   module.forward=types.MethodType(forward,module)
 if prepared and cached is None:write(prepared,payloads)
 metadata_bytes=(Path(prepared)/'layout.json').stat().st_size if prepared else 0
 return metadata_bytes+storage_bytes(payloads)+sum(p.numel()*p.element_size() for p in model.parameters())


def install_execution_policy(model,hooks):
 for i,layer in enumerate(model.model.layers):
  original=layer.forward
  def forward(self,hidden_states,*args,index=i,original=original,**kwargs):
   summary=torch.stack((hidden_states.float().square().mean(),hidden_states.float().abs().max()))
   decision=hooks.execute_layer(summary,index,len(model.model.layers))
   if type(decision)is not bool:raise ValueError('execution policy returns a bool')
   return original(hidden_states,*args,**kwargs) if decision else hidden_states
  layer.forward=types.MethodType(forward,layer)


def generate_verified(model,hooks,prompt,count):
 tokens=prompt;history=[];result=[]
 initial=model(prompt,use_cache=True,logits_to_keep=1);cache=initial.past_key_values;next_logits=initial.logits[:,-1:]
 while len(result)<count:
  proposal=hooks.propose(tokens.clone(),history,min(16,count-len(result)))
  if not isinstance(proposal,torch.Tensor) or proposal.dtype!=torch.int64 or proposal.ndim!=2 or proposal.shape[0]!=tokens.shape[0] or proposal.shape[1]>min(16,count-len(result)):raise ValueError('invalid draft proposal')
  if (proposal<0).any() or (proposal>=model.config.vocab_size).any():raise ValueError('draft token out of vocabulary')
  if storage_bytes(proposal)+storage_bytes(history)>4*1024**3:raise ValueError('auxiliary memory exceeds cap')
  proposal=proposal.to(tokens.device);prior=cache.get_seq_length()
  if proposal.shape[1]:
   verification=model(proposal,past_key_values=cache,use_cache=True)
   target=torch.cat((next_logits,verification.logits),1);cache=verification.past_key_values
  else:target=next_logits
  accepted=0
  for j in range(proposal.shape[1]):
   if not torch.equal(target[:,j].argmax(-1,keepdim=True),proposal[:,j:j+1]):break
   accepted+=1
  cache.crop(prior+accepted);additions=proposal[:,:accepted]
  if len(result)+accepted<count:
   extra=target[:,accepted].argmax(-1,keepdim=True);additions=torch.cat((additions,extra),-1)
   advanced=model(extra,past_key_values=cache,use_cache=True,logits_to_keep=1);cache=advanced.past_key_values;next_logits=advanced.logits[:,-1:]
  tokens=torch.cat((tokens,additions),-1);result.extend(additions.transpose(0,1));history.append(accepted)
 return tokens[:,prompt.shape[1]:]


def python_tool(code):
 # Restricted Python arithmetic expression service: no imports, filesystem,
 # network, reflection, or object access. The child also has OS resource caps.
 import subprocess,sys
 source='''import ast,resource,os,sys
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
tree=ast.parse(sys.stdin.read(),mode='eval')
allowed=(ast.Expression,ast.Constant,ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,ast.USub,ast.UAdd)
if any(not isinstance(n,allowed) for n in ast.walk(tree)):raise ValueError('arithmetic expressions only')
print(eval(compile(tree,'<tool>','eval'),{'__builtins__':{}},{}))
'''
 return subprocess.run([sys.executable,'-I','-c',source],input=code,text=True,capture_output=True,timeout=3,check=True).stdout.strip()


def _reason(model,tok,hooks,question,device,smoke=False,ranker=None):
 begin=time.monotonic();history=[];spent=0
 for step in range(32):
  remaining={'seconds':max(0.,30-(time.monotonic()-begin)),'tokens':8192-spent}
  if remaining['seconds']<=0:raise ValueError('answer production exceeded 30 seconds')
  action,payload=hooks.plan(question,history,remaining)
  if action=='answer':
   if time.monotonic()-begin>30:raise ValueError('answer production exceeded 30 seconds')
   return payload,spent
  if action=='python':history.append({'answer':python_tool(payload),'response':'tool'})
  elif action=='sample':
   if type(payload)is not int or not 1<=payload<=remaining['tokens']:raise ValueError('generated-token budget')
   if smoke:history.append({'answer':'4','response':'2 + 2 = 4','score':ranker('2 + 2 = 4') if ranker else 0.});spent+=1;continue
   x=tok.apply_chat_template([{'role':'user','content':question}],add_generation_prompt=True,return_tensors='pt').to(device)
   with torch.no_grad():seq=model.generate(x,max_new_tokens=payload,max_time=max(.01,remaining['seconds']-2.),do_sample=True,temperature=.8,pad_token_id=tok.eos_token_id)
   spent+=seq.shape[1]-x.shape[1];text=tok.decode(seq[0,x.shape[1]:],skip_special_tokens=True)
   history.append({'answer':answer(text),'response':text,'score':ranker(text) if ranker else 0.})
  else:raise ValueError('unknown answer action')
 raise ValueError('no answer within 32 service actions')


def reason(model,tok,hooks,question,device,smoke=False,ranker=None):
 import signal
 def expired(signum,frame):raise ValueError('answer production exceeded 30 seconds')
 previous=signal.signal(signal.SIGALRM,expired)
 signal.setitimer(signal.ITIMER_REAL,30.)
 try:return _reason(model,tok,hooks,question,device,smoke,ranker)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0.);signal.signal(signal.SIGALRM,previous)
