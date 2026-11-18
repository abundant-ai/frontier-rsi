"""Arrival-driven serving with externally owned clocks and request accounting."""
import math
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from research_model import storage_bytes
from inference_core import sync


def batch_step(model,entries,device):
 # A single request needs no batch padding, cache repacking, or per-layer
 # gather. Reuse its ordinary contiguous cache through the fixed model.
 if len(entries)==1:
  tokens,state,position=entries[0];past=0 if state is None else state[0][0].shape[-2]
  ids=torch.as_tensor(tokens,device=device)[None,:]
  cache=None if state is None else DynamicCache(ddp_cache_data=state)
  positions=torch.arange(position,position+len(tokens),device=device)
  result=model(ids,past_key_values=cache,position_ids=positions[None,:],cache_position=torch.arange(past,past+len(tokens),device=device),use_cache=True,logits_to_keep=1)
  state=[(layer.keys,layer.values) for layer in result.past_key_values.layers]
  return [(int(result.logits[0,-1].argmax()),state)]
 return batch_step_general(model,entries,device)


def batch_step_general(model,entries,device):
 # Each entry is (new tokens, previous per-layer key/value tensors, absolute
 # position). Left-pad cached context and right-pad current computation; masks
 # prevent either padding region from becoming a model dependency.
 b=len(entries);qmax=max(len(e[0]) for e in entries);past=max(0 if e[1] is None else e[1][0][0].shape[-2] for e in entries)
 ids=torch.zeros(b,qmax,dtype=torch.long,device=device);positions=torch.zeros_like(ids)
 mask=torch.full((b,1,qmax,past+qmax),float('-inf'),device=device,dtype=model.dtype)
 states=[]
 for layer in range(model.config.num_hidden_layers):
  keys=[];values=[]
  for tokens,state,position in entries:
   shape=(1,model.config.num_key_value_heads,past,model.config.hidden_size//model.config.num_attention_heads)
   k=torch.zeros(shape,device=device,dtype=model.dtype);v=torch.zeros_like(k)
   if state is not None:
    length=state[layer][0].shape[-2];k[...,-length:,:]=state[layer][0];v[...,-length:,:]=state[layer][1]
   keys.append(k);values.append(v)
  states.append((torch.cat(keys),torch.cat(values)))
 cache=DynamicCache(ddp_cache_data=states)
 for i,(tokens,state,position) in enumerate(entries):
  t=len(tokens);n=0 if state is None else state[0][0].shape[-2]
  ids[i,:t]=torch.as_tensor(tokens,device=device);positions[i,:t]=torch.arange(position,position+t,device=device)
  if n:mask[i,0,:t,past-n:past]=0
  mask[i,0,:t,past:past+t]=torch.where(torch.ones(t,t,device=device,dtype=torch.bool).tril(),0.,float('-inf'))
  # Unused query rows receive one harmless dependency to avoid NaN softmax.
  if t<qmax:mask[i,0,t:,past]=0
 selected=sorted(set(len(entry[0])-1 for entry in entries))
 result=model(ids,past_key_values=cache,attention_mask=mask,position_ids=positions,cache_position=torch.arange(past,past+qmax,device=device),use_cache=True,logits_to_keep=torch.tensor(selected,device=device))
 outputs=[]
 for i,(tokens,state,position) in enumerate(entries):
  t=len(tokens);n=0 if state is None else state[0][0].shape[-2]
  saved=[]
  for layer in result.past_key_values.layers:
   index=torch.cat((torch.arange(past-n,past,device=device),torch.arange(past,past+t,device=device)))
   saved.append((layer.keys[i:i+1].index_select(-2,index).contiguous(),layer.values[i:i+1].index_select(-2,index).contiguous()))
  outputs.append((int(result.logits[i,selected.index(t-1)].argmax()),saved))
 return outputs


def run_trace(model,hooks,requests,rate,device,row,smoke=False,replay_plan=None):
 # Idle intervals advance the trace clock; GPU work advances it by measured
 # wall time. No self-reported candidate timers or fabricated model work.
 clock=0.;next_request=0;active={};finished={};first=[];inter=[];peak=0;gaps={};execution_plan=[];plan_index=0;replay_compute_seconds=0.
 trace=[dict(r,arrival=i/rate) for i,r in enumerate(requests)]
 while next_request<len(trace) or active:
  recorded=None
  if replay_plan is not None:
   if plan_index>=len(replay_plan):raise ValueError('reference replay plan ended before all requests completed')
   recorded=replay_plan[plan_index];start=float(recorded['started_at']);end=float(recorded['finished_at'])
   if not math.isfinite(start) or not math.isfinite(end) or start<clock or end<start:raise ValueError('invalid replay clock')
   clock=start
  elif not active:clock=max(clock,trace[next_request]['arrival'])
  # Requests beyond capacity wait in the arrival queue. Their original
  # arrival time is retained for latency and no request is discarded.
  while next_request<len(trace) and trace[next_request]['arrival']<=clock and len(active)<64:
   r=trace[next_request];active[next_request]={'prompt':r['input'],'count':r['output_length'],'arrival':r['arrival'],'offset':0,'generated':[],'cache':None,'last':None};next_request+=1
  summary=[{'id':i,'remaining':len(r['prompt'])-r['offset'],'generated':len(r['generated']),'age':clock-r['arrival']} for i,r in active.items()]
  import time
  sync(device);begin=time.perf_counter()
  chosen=recorded['chosen'] if recorded is not None else hooks.schedule(summary,clock)
  if not isinstance(chosen,(list,tuple)) or not chosen or len(chosen)>64:raise ValueError('scheduler must advance a nonempty request batch')
  ids=[item[0] for item in chosen]
  if len(set(ids))!=len(ids) or any(type(i)is not int or i not in active for i in ids):raise ValueError('invalid/duplicate scheduled request')
  step_start=clock
  entries=[]
  for i,n in chosen:
   r=active[i];remaining=len(r['prompt'])-r['offset']
   if type(n)is not int or n<1 or n>8192 or not remaining and n!=1:raise ValueError('invalid scheduled token count')
   tokens=r['prompt'][r['offset']:r['offset']+n] if remaining else [r['generated'][-1]]
   cache=r['cache']
   if row==55 and cache is not None:cache=[(hooks.decode(k).to(device=device,dtype=model.dtype),hooks.decode(v).to(device=device,dtype=model.dtype)) for k,v in cache]
   entries.append((tokens,cache,r['offset']+max(0,len(r['generated'])-1)))
  with torch.no_grad():outputs=batch_step(model,entries,device)
  if row==55:outputs=[(token,[(hooks.encode(k),hooks.encode(v)) for k,v in state]) for token,state in outputs]
  sync(device);duration=time.perf_counter()-begin
  if recorded is None:
   clock+=duration
   execution_plan.append({'started_at':step_start,'finished_at':clock,'chosen':[[i,n] for i,n in chosen]})
  else:
   replay_compute_seconds+=duration;clock=end;plan_index+=1
  for (i,n),(token,state),(tokens,old,position) in zip(chosen,outputs,entries):
   r=active[i];r['offset']+=len(tokens) if r['offset']<len(r['prompt']) else 0;r['cache']=state
   if r['offset']==len(r['prompt']):
    if r['last'] is None:first.append(clock-r['arrival'])
    else:inter.append(clock-r['last'])
    r['last']=clock;r['generated'].append(token)
    if len(r['generated'])==r['count']:finished[i]=r['generated'];del active[i]
  del entries,outputs
  peak=max(peak,storage_bytes([r['cache'] for r in active.values()]))
  if not math.isfinite(clock):raise ValueError('nonfinite serving clock')
 if len(finished)!=len(requests):raise ValueError('request dropped')
 if replay_plan is not None and plan_index!=len(replay_plan):raise ValueError('reference replay has unused plan steps')
 p95=lambda xs:float(np.quantile(xs,.95)) if xs else 0.
 result={'elapsed':clock,'ttft_p95':p95(first),'intertoken_p95':p95(inter),'state_bytes':peak,'requests':len(finished),'output_tokens':sum(map(len,finished.values())),'outputs':[finished[i] for i in range(len(trace))]}
 if replay_plan is None:result['execution_plan']=execution_plan
 else:result.update(reference_replay_only=True,replay_compute_seconds=replay_compute_seconds,replay_plan_steps=plan_index)
 return result
