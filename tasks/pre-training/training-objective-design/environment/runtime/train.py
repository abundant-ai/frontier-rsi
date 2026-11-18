"""Fixed training runner for the submitted research hook."""
import argparse,contextlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from model import GPT,Config,token_loss
from policy import load

RUN_DEADLINE = float('inf')


def checkpoint(module,model,out):
 temporary=out/'checkpoint.tmp'
 export(module,model,temporary)
 temporary.replace(out/'model.safetensors')


def logprob(model,ids,labels):
 logits=model(input_ids=ids).logits[:,:-1].float()
 target=labels[:,1:];mask=target!=-100
 safe=target.clamp_min(0)
 return (logits.log_softmax(-1).gather(-1,safe[...,None]).squeeze(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)


def encode_pair(tok,messages,device,limit=2048):
 ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)[:limit]
 prompt=tok.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True)
 x=torch.tensor([ids],device=device);labels=x.clone();labels[:,:min(len(prompt),len(ids)-1)]=-100
 return x,labels


def export(module,model,path):
 if getattr(module,'bounded_adapter',False):
  from adapter import export as trusted_export
  state=trusted_export(model)
 elif getattr(module,'bounded_objective',False):
  from training_objective import export as trusted_export
  state=trusted_export(model)
 else:state=module.export(model) if hasattr(module,'export') else {k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}
 if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) and v.is_floating_point() and torch.isfinite(v).all() for v in state.values()):raise ValueError('invalid exported state')
 save_file(state,str(path))


def text_train(c,m,seconds,seed,device,assets,out,smoke):
 cfg=Config(vocab=64,width=32,layers=2,heads=4,context=64) if smoke else Config()
 model=GPT(cfg).to(device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 if smoke:
  tokens=np.random.default_rng(seed).integers(0,64,size=8192,dtype=np.int32);docs=np.array([(i,64) for i in range(0,8192,64)])
  metadata=torch.linspace(.1,1,len(docs));buckets=np.arange(len(docs))%8
 else:
  tokens=np.load(assets/'train.npy',mmap_mode='r');docs=np.load(assets/'documents.npy')
  metadata=torch.from_numpy(np.load(assets/'metadata.npy'));buckets=np.load(assets/'buckets.npy')
 mode=c['mode']
 if mode=='corpus':
  selection=m.prepare_corpus(tokens,docs,metadata)
  selection=torch.as_tensor(selection).detach().cpu()
  if selection.dtype!=torch.long or selection.ndim!=1 or not len(selection) or (selection<0).any() or (selection>=len(docs)).any():raise ValueError('corpus selection must be nonempty int64 document indices')
  docs=docs[selection.numpy()]
 trainable=tuple(model.parameters());auxiliary=()
 if mode=='objective':
  import training_objective
  trainable,auxiliary,total_parameters=training_objective.prepare(model,m,1_000_000 if smoke else 250_000_000)
  forward=lambda ids:training_objective.forward(model,ids)
 gen=torch.Generator().manual_seed(seed)
 opt=m.optimizer(model) if mode=='updates' else torch.optim.AdamW(trainable,lr=6e-4,weight_decay=.1)
 original_lrs=[g['lr'] for g in opt.param_groups]
 start=time.monotonic();last_checkpoint=start;steps=0;limit=c.get('smoke_steps',2) if smoke else None
 context=16 if smoke else 1024;micro=2 if smoke else 8;accum=1 if smoke else 8
 while (steps<limit if smoke else time.monotonic()<RUN_DEADLINE-15):
  opt.zero_grad(set_to_none=True);progress=min(1.,(time.monotonic()-start)/seconds)
  if mode=='rate':
   for group,base in zip(opt.param_groups,original_lrs):group['lr']=float(m.learning_rate(progress,base))
  for _ in range(accum):
   if mode=='sampling':idx=m.select_examples(metadata,micro,gen,progress)
   elif mode=='sources':
    probs=m.source_probabilities(progress,8).detach().cpu()
    if probs.shape!=(8,) or not torch.isfinite(probs).all() or (probs<0).any() or probs.sum()<=0:raise ValueError('source probabilities')
    sources=torch.multinomial(probs,micro,replacement=True,generator=gen)
    idx=torch.tensor([int(np.flatnonzero(buckets==int(v))[int(torch.randint(int((buckets==int(v)).sum()),(1,),generator=gen))]) for v in sources])
   else:idx=torch.randint(len(docs),(micro,),generator=gen)
   if len(idx)!=micro or (idx<0).any() or (idx>=len(docs)).any():raise ValueError('document index contract')
   x=[];y=[]
   for i in idx.tolist():
    offset,n=map(int,docs[i]);n=min(n,context+1)
    z=torch.tensor(np.array(tokens[offset:offset+n]),dtype=torch.long)
    if n<2:raise ValueError('document too short')
    # Padding excludes targets and cannot introduce cross-document training targets.
    xx=torch.zeros(context,dtype=torch.long);yy=torch.full((context,),-100,dtype=torch.long)
    xx[:n-1]=z[:-1];yy[:n-1]=z[1:];x.append(xx);y.append(yy)
   x=torch.stack(x).to(device);y=torch.stack(y).to(device)
   ctx=m.numerical_context(device) if mode=='numerics' else (torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext())
   with ctx:loss=(m.training_loss(forward,x,y,auxiliary) if mode=='objective' else token_loss(model(x),y))/accum
   if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
   loss.backward()
  torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step();steps+=1
  if not smoke and time.monotonic()-last_checkpoint>=60:
   checkpoint(m,model,out);last_checkpoint=time.monotonic()
 if steps==0:raise ValueError('no completed training updates')
 checkpoint(m,model,out)
 result={'steps':steps,'config':vars(cfg),'elapsed':time.monotonic()-start}
 if mode=='objective':result.update(training_parameters=total_parameters,auxiliary_parameters=sum(p.numel() for p in auxiliary))
 return result


def post_train(c,m,seconds,seed,device,assets,out,smoke):
 from transformers import AutoModelForCausalLM,AutoTokenizer
 if smoke:return post_smoke(c,m,device,out)
 tok=AutoTokenizer.from_pretrained(assets/'model',local_files_only=True);tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(assets/'model',local_files_only=True,torch_dtype=torch.bfloat16 if device=='cuda' else torch.float32,attn_implementation='sdpa').to(device)
 model.config.use_cache=False;mode=c['mode']
 if mode=='adapt':
  import adapter
  trainable,settings=adapter.prepare(model,m)
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  model.enable_input_require_grads()
  opt=torch.optim.AdamW(trainable,lr=settings['lr'],weight_decay=settings['weight_decay'])
 else:
  trainable=tuple(p for p in model.parameters() if p.requires_grad)
  if not trainable:raise ValueError('no trainable parameters')
  opt=torch.optim.AdamW(trainable,lr=1e-5)
 start=time.monotonic();last_checkpoint=start;steps=0
 if mode in ('preferences','adapt'):
  rows=json.loads((assets/'pref_train.json').read_text())
  reference=None
  if mode=='preferences':
   reference=AutoModelForCausalLM.from_pretrained(assets/'model',local_files_only=True,torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
   # Explicit reward estimator on fixed per-response features, fit from preference pairs.
   reward=torch.nn.Linear(2,1).to(device);critic=torch.nn.Linear(2,1).to(device)
   reward_opt=torch.optim.AdamW(list(reward.parameters())+list(critic.parameters()),lr=.01)
  while time.monotonic()<RUN_DEADLINE-30:
   row=rows[steps%len(rows)];chosen,cl=encode_pair(tok,row['chosen'],device);rejected,rl=encode_pair(tok,row['rejected'],device)
   opt.zero_grad(set_to_none=True)
   if mode=='adapt':loss=-logprob(model,chosen,cl).mean()
   else:
    with torch.no_grad():rc=logprob(reference,chosen,cl);rr=logprob(reference,rejected,rl)
    fc=torch.stack((rc,torch.tensor([chosen.shape[1]/2048],device=device)),-1);fr=torch.stack((rr,torch.tensor([rejected.shape[1]/2048],device=device)),-1)
    reward_opt.zero_grad();reward_fit=-F.logsigmoid(reward(fc)-reward(fr)).mean()
    target=(reward(fc).detach()+reward(fr).detach())/2
    reward_fit=reward_fit+F.mse_loss(critic((fc+fr)/2),target);reward_fit.backward();reward_opt.step()
    with torch.no_grad():oldc=logprob(model,chosen,cl);oldr=logprob(model,rejected,rl);old=torch.stack((oldc,oldr),-1).log_softmax(-1)
    pc=logprob(model,chosen,cl);pr=logprob(model,rejected,rl)
    loss=m.preference_loss(pc,pr,rc,rr,(reward(fc).detach().squeeze(-1),reward(fr).detach().squeeze(-1),old))
   if not torch.isfinite(loss):raise ValueError('nonfinite objective')
   loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step();steps+=1
   if time.monotonic()-last_checkpoint>=60:
    checkpoint(m,model,out);last_checkpoint=time.monotonic()
 else:
  rows=json.loads((assets/'math_train.json').read_text());critic=torch.nn.Linear(model.config.hidden_size,1).to(device,dtype=model.dtype)
  opt=torch.optim.AdamW(list(model.parameters())+list(critic.parameters()),lr=1e-6)
  summary=torch.zeros(len(rows),3);summary[:,1]=.5;summary[:,2]=torch.tensor([len(r['question']) for r in rows])
  generator=torch.Generator().manual_seed(seed);generated_tokens=0
  while time.monotonic()<RUN_DEADLINE-30:
   count,max_tokens,index=4,512,steps%len(rows)
   if mode=='allocation':
    allocation=m.allocate(summary.clone(),min(1.,(time.monotonic()-start)/seconds),generator)
    if not isinstance(allocation,(tuple,list)) or len(allocation)!=3 or any(type(v) is not int for v in allocation):raise ValueError('allocate returns integer prompt index, response count, and token limit')
    index,count,max_tokens=allocation
    if not 0<=index<len(rows) or not 1<=count<=16 or not 1<=max_tokens<=2048:raise ValueError('allocation outside data-collection limits')
    if generated_tokens+count*max_tokens>10_000_000:break
   row=rows[index];prompt=tok.apply_chat_template([{'role':'user','content':row['question']}],tokenize=False,add_generation_prompt=True)
   x=tok(prompt,return_tensors='pt',truncation=True,max_length=1024).input_ids.to(device).repeat(count,1)
   model.eval()
   with torch.no_grad():seq=model.generate(x,max_new_tokens=max_tokens,do_sample=True,temperature=.8,pad_token_id=tok.eos_token_id,use_cache=True)
   generated_tokens+=count*(seq.shape[1]-x.shape[1])
   labels=seq.clone();labels[:,:x.shape[1]]=-100
   texts=tok.batch_decode(seq[:,x.shape[1]:],skip_special_tokens=True)
   rewards=torch.tensor([float(answer(t)==answer(row['answer'])) for t in texts],device=device)
   visits=float(summary[index,0]);summary[index,1]=(summary[index,1]*visits+float(rewards.sum()))/(visits+count);summary[index,0]+=count
   with torch.no_grad():old=logprob(model,seq,labels)
   model.train();opt.zero_grad(set_to_none=True)
   result=model(input_ids=seq,output_hidden_states=True);features=result.hidden_states[-1][:,-1]
   values=critic(features.detach()).float().squeeze(-1)
   logits=result.logits[:,:-1].float();target=labels[:,1:];mask=target!=-100
   lp=(logits.log_softmax(-1).gather(-1,target.clamp_min(0)[...,None]).squeeze(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
   loss=fixed_policy_loss(lp,old,rewards,values) if mode=='allocation' else m.policy_loss(lp,old,rewards,values)
   if not torch.isfinite(loss):raise ValueError('nonfinite objective')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
   if time.monotonic()-last_checkpoint>=60:
    checkpoint(m,model,out);last_checkpoint=time.monotonic()
 if steps==0:raise ValueError('no completed training updates')
 checkpoint(m,model,out)
 result={'steps':steps,'elapsed':time.monotonic()-start}
 if mode=='adapt':result['trainable_parameters']=settings['trainable_parameters']
 return result


def fixed_policy_loss(logprobs,old_logprobs,rewards,values):
 advantages=rewards-values.detach();ratio=(logprobs-old_logprobs).exp()
 return -torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages).mean()+.5*(values-rewards).square().mean()


def answer(text):
 import re
 matches=re.findall(r'[-+]?\d+(?:\.\d+)?',text.replace(',',''))
 return matches[-1].lstrip('+') if matches else None


def post_smoke(c,m,device,out):
 # Exercise every hook with actual differentiable tensors without downloading a model.
 if c['mode']=='adapt':
  import adapter
  model=torch.nn.Module();model.mlp=torch.nn.Module();model.mlp.down_proj=torch.nn.Linear(8,8).to(device)
  model.q_proj=torch.nn.Linear(8,8).to(device);model.v_proj=torch.nn.Linear(8,8).to(device)
  trainable,settings=adapter.prepare(model,m)
  opt=torch.optim.AdamW(trainable,lr=settings['lr'])
  for _ in range(2):
   opt.zero_grad();model.mlp.down_proj(model.q_proj(model.v_proj(torch.randn(4,8,device=device)))).square().mean().backward();opt.step()
  save_file(adapter.export(model),str(out/'model.safetensors'))
  return {'steps':2,'smoke_only':True}
 model=torch.nn.Linear(8,8).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.01)
 for _ in range(2):
  opt.zero_grad();v=model(torch.randn(4,8,device=device));lp=v.log_softmax(-1).mean(-1)
  if c['mode']=='preferences':loss=m.preference_loss(lp,lp-.1,lp.detach(),lp.detach()-.1,(torch.ones(4),torch.zeros(4),torch.stack((lp,lp-.1),-1).detach().log_softmax(-1)))
  elif c['mode']=='allocation':
   allocation=m.allocate(torch.tensor([[0.,.5,10.],[4.,.25,20.]]),.5,torch.Generator().manual_seed(17))
   if len(allocation)!=3:raise ValueError('allocation shape')
   loss=fixed_policy_loss(lp,lp.detach(),torch.tensor([0.,1.,0.,1.]),torch.zeros(4,requires_grad=True))
  elif c['mode']=='policy':loss=m.policy_loss(lp,lp.detach(),torch.tensor([0.,1.,0.,1.]),torch.zeros(4,requires_grad=True))
  else:loss=v.square().mean()
  loss.backward();opt.step()
 save_file({k:v.detach().clone() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
 return {'steps':2,'smoke_only':True}


def validate_duration(contract,seconds,smoke=False):
 if smoke:return
 allowed=120<=seconds<=1800
 if not allowed:raise ValueError('run duration outside contract')


def main():
 global RUN_DEADLINE
 p=argparse.ArgumentParser();p.add_argument('--config',default='/app/contract.json');p.add_argument('--candidate',default='/app/submission/candidate.py');p.add_argument('--assets',default='/opt/assets/public');p.add_argument('--out',required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 c=json.loads(Path(a.config).read_text())
 validate_duration(c,a.seconds,a.smoke)
 RUN_DEADLINE=time.monotonic()+a.seconds
 if c['mode']=='adapt':
  import adapter
  m=adapter.load(a.candidate,c['entrypoints']);m.bounded_adapter=True
 elif c['mode']=='objective':
  import training_objective
  m=training_objective.load(a.candidate,c['entrypoints']);m.bounded_objective=True
 else:m=load(a.candidate,c['entrypoints'])
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(min(1.,c['memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 start=time.monotonic()
 result=(text_train if c['family']=='text' else post_train)(c,m,a.seconds,a.seed,a.device,Path(a.assets),out,a.smoke)
 result['total_elapsed']=time.monotonic()-start
 if a.device=='cuda':result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
 (out/'run.json').write_text(json.dumps(result))
if __name__=='__main__':main()
