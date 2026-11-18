from policy import fresh
from measurement_protocol import measurement_policies, aggregate
"""Two-GPU prefill/decode execution with measured KV transfers and overlap."""
import gc
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from policy import CandidateError,call
from prefix_service import cache_object,execute,generate,requests,sync,token_bytes


def workload(tokenizer,seed,quick=False,smoke=False):
    prompts=requests(tokenizer,seed,quick=False,smoke=smoke)
    result=[]
    rng=random.Random(seed)
    for i in range(6 if quick or smoke else 24):
        length=(12,24)[i%2] if smoke else (512,2048,4096)[i%3]
        prompt=prompts[i%len(prompts)]
        prompt=tuple((list(prompt)*((length+len(prompt)-1)//len(prompt)))[:length])
        count=3 if smoke else 4 if quick else (16,32)[i%2]
        arrival=0.0 if i<4 else (i-3)*(.0001 if smoke else .03)+rng.uniform(0,.01 if not smoke else .00001)
        result.append((prompt,arrival,count))
    return tuple(result)


def state_bytes(state):
    return 0 if state is None else sum(t.numel()*t.element_size() for pair in state['cache'] for t in pair)


def reserve(row,quantum,bpt):
    # Peak request-owned state during this action, before its final release.
    generated,required=row[2],row[3]
    steps=quantum-(generated+quantum==required)
    return max(row[5],(row[1]+generated+steps-1)*bpt if steps else 0)


def checked_actions(policy,now,ready,busy,free,bpt):
    actions=call(policy,now,ready,busy,free,bpt)
    if not isinstance(actions,(tuple,list)) or len(actions)>2:
        raise CandidateError('plan must return up to two (gpu, request_ids, quantum) actions')
    rows={row[0]:row for row in ready}
    devices=set();selected=set();checked=[]
    for action in actions:
        if not isinstance(action,(tuple,list)) or len(action)!=3:
            raise CandidateError('invalid execution action')
        gpu,ids,quantum=action
        if type(gpu) is not int or gpu not in (0,1) or busy[gpu] or gpu in devices:
            raise CandidateError('select an idle GPU at most once')
        if not isinstance(ids,(tuple,list)) or not ids or len(ids)>8 or any(type(i) is not int or i not in rows for i in ids):
            raise CandidateError('select 1–8 arrived ready requests')
        if len(set(ids))!=len(ids) or set(ids)&selected:
            raise CandidateError('request selected more than once')
        if type(quantum) is not int or quantum<1 or any(quantum>rows[i][3]-rows[i][2] for i in ids):
            raise CandidateError('invalid output quantum')
        if len({(rows[i][2]==0,rows[i][1]+max(0,rows[i][2]-1)) for i in ids})!=1:
            raise CandidateError('batch requires equal prefill lengths or decode cache lengths')
        extra=sum(reserve(rows[i],quantum,bpt)-rows[i][5] for i in ids)
        if extra>free:
            raise CandidateError('execution actions exceed the 2 GiB persistent-state reservation')
        free-=extra
        devices.add(gpu);selected.update(ids)
        checked.append((gpu,tuple(ids),quantum))
    if ready and not checked and not any(busy):
        raise CandidateError('cannot idle both GPUs while requests are ready')
    return checked


@torch.inference_mode()
def execute_batch(model,device,gpu,ids,quantum,jobs,old_states,outputs,started):
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)
    transferred=0
    states=[]
    for old in old_states:
        if old is None:
            states.append(None)
        elif old['gpu']==gpu:
            states.append(old['cache'])
        else:
            transferred+=state_bytes(old)
            states.append(tuple(tuple(t.to(device=device,copy=True) for t in pair) for pair in old['cache']))
    generated=[[] for _ in ids]
    times=[[] for _ in ids]
    previous=[list(row) for row in outputs]
    for _ in range(quantum):
        if states[0] is None:
            inputs=torch.tensor([jobs[i][0] for i in ids],dtype=torch.long,device=device)
            past=None
        else:
            inputs=torch.tensor([[row[-1]] for row in previous],dtype=torch.long,device=device)
            past=tuple((torch.cat([state[layer][0] for state in states]),
                        torch.cat([state[layer][1] for state in states]))
                       for layer in range(model.config.num_hidden_layers))
        result=model(inputs,past_key_values=cache_object(past),use_cache=True,logits_to_keep=1)
        new_states=[]
        for row,i in enumerate(ids):
            if len(previous[row])+1==jobs[i][2]:
                new_states.append(None)
            else:
                new_states.append(tuple((layer.keys[row:row+1].clone(),layer.values[row:row+1].clone())
                                        for layer in result.past_key_values.layers))
        token_ids=result.logits[:,-1].argmax(-1).cpu().tolist()
        completed=time.perf_counter()-started
        for row,token in enumerate(token_ids):
            previous[row].append(token);generated[row].append(token);times[row].append(completed)
        states=new_states
        del result,inputs,past
    sync(device)
    return generated,times,[None if state is None else {'gpu':gpu,'cache':state} for state in states],transferred


def run_trace(models,devices,policy,jobs,max_seconds=120):
    for device in devices:
        sync(device)
    started=time.perf_counter()
    policy=fresh(policy)
    states=[None for _ in jobs]
    outputs=[[] for _ in jobs]
    timestamps=[[] for _ in jobs]
    complete=set();inflight={};reserved={}
    budget=2*1024**3;bpt=token_bytes(models[0]);peak_reserved=0;transfers=0;dispatches=0
    for device in devices:
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
    pools=[ThreadPoolExecutor(max_workers=1) for _ in devices]
    try:
        while len(complete)<len(jobs):
            now=time.perf_counter()-started
            if now>max_seconds:
                raise CandidateError('all generation requests must finish within 120 seconds')
            for gpu,(future,ids) in tuple(inflight.items()):
                if not future.done():
                    continue
                tokens,times,new_states,moved=future.result()
                transfers+=moved
                for i,new,ts,state in zip(ids,tokens,times,new_states):
                    outputs[i].extend(new);timestamps[i].extend(ts);states[i]=state
                    reserved.pop(i)
                    if len(outputs[i])==jobs[i][2]:
                        complete.add(i)
                del inflight[gpu]
                del tokens,times,new_states,state,future
            ready=tuple((i,len(prompt),len(outputs[i]),count,-1 if states[i] is None else states[i]['gpu'],
                         state_bytes(states[i]),arrival)
                        for i,(prompt,arrival,count) in enumerate(jobs)
                        if arrival<=now and i not in complete and i not in reserved)
            used=sum(state_bytes(state) for state in states)+sum(reserved.values())
            peak_reserved=max(peak_reserved,used)
            busy=tuple(gpu in inflight for gpu in range(2))
            if ready and not all(busy):
                actions=checked_actions(policy,now,ready,busy,budget-used,bpt)
                rows={row[0]:row for row in ready}
                for gpu,ids,quantum in actions:
                    old=[states[i] for i in ids]
                    for i in ids:
                        reserved[i]=reserve(rows[i],quantum,bpt)
                        states[i]=None
                    peak_reserved=max(peak_reserved,sum(state_bytes(state) for state in states)+sum(reserved.values()))
                    future=pools[gpu].submit(execute_batch,models[gpu],devices[gpu],gpu,ids,quantum,jobs,old,
                                             [tuple(outputs[i]) for i in ids],started)
                    inflight[gpu]=(future,ids);dispatches+=1
                    del old,future
                del actions
            if len(complete)<len(jobs):
                time.sleep(.001)
    finally:
        for pool in pools:
            pool.shutdown(wait=True,cancel_futures=True)
    elapsed=time.perf_counter()-started
    first=[row[0]-jobs[i][1] for i,row in enumerate(timestamps)]
    gaps=[b-a for row in timestamps for a,b in zip(row,row[1:])]
    p95=lambda values:sorted(values)[math.ceil(.95*len(values))-1]
    peaks=[torch.cuda.max_memory_allocated(d) if str(d).startswith('cuda') else 0 for d in devices]
    del states
    gc.collect()
    return {'outputs':outputs,'elapsed_seconds':elapsed,'p95_first_token_seconds':p95(first),
            'p95_inter_token_seconds':p95(gaps),'peak_gpu_bytes':peaks,'peak_reserved_bytes':peak_reserved,
            'cross_device_kv_bytes':transfers,'dispatches':dispatches,'token_timestamps':timestamps}


def measured_trace(models,devices,policy,jobs,role):
    if role not in ('candidate','baseline'):
        raise RuntimeError('unknown measurement role')
    try:
        return run_trace(models,devices,policy,jobs)
    except CandidateError as exc:
        if role=='baseline':
            raise RuntimeError('fixed paired baseline execution failed: '+str(exc)) from exc
        raise


def validate_measured_trace(row,expected,role,repetition):
    if role not in ('candidate','baseline'):
        raise RuntimeError('unknown measurement role')
    # A failed control execution aborts the measurement rather than producing
    # a policy-specific error.
    error=RuntimeError if role=='baseline' else CandidateError
    prefix='fixed paired baseline: ' if role=='baseline' else 'candidate: '
    if row.pop('outputs')!=expected:
        raise error(prefix+'prefill/decode generation differs from independent full-request execution')
    if any(peak>16*1024**3 for peak in row['peak_gpu_bytes']):
        raise error(prefix+'GPU allocation exceeds 16 GiB on a device')
    if row['peak_reserved_bytes']>2*1024**3:
        raise error(prefix+'persistent-state reservation exceeds 2 GiB')
    # Latency limits apply to candidate executions; control executions retain
    # their actual timing even when slower than these limits.
    if role=='candidate' and (row['p95_first_token_seconds']>20 or row['p95_inter_token_seconds']>2):
        raise CandidateError('candidate p95 latency limit exceeded: first_token='
                             +str(row['p95_first_token_seconds'])+'s (limit 20s), inter_token='
                             +str(row['p95_inter_token_seconds'])+'s (limit 2s), repetition='+str(repetition))


@torch.inference_mode()
def evaluate_pipeline(models,devices,policy,jobs,quick=False,*,repetitions=None,baseline=None,pair_offset=0):
    expected=[]
    for prompt,_,count in jobs:
        out=execute(models[0],prompt,None,count,devices[0])
        expected.append(generate(models[0],out,count,devices[0]))
        del out
    for model,device in zip(models,devices):
        out=execute(model,jobs[0][0],None,1,device)
        generate(model,out,2,device);sync(device);del out
    measured=[]
    for repetition,role,current in measurement_policies(policy,lambda original:original,quick=quick,
                                                         repetitions=repetitions,baseline=baseline,pair_offset=pair_offset):
        row=measured_trace(models,devices,current,jobs,role)
        row.update(repetition=repetition,role=role)
        validate_measured_trace(row,expected,role,repetition)
        measured.append(row)
    return {**aggregate(measured,'elapsed_seconds'),'verified_requests':len(jobs),
            'verified_output_tokens':sum(job[2] for job in jobs)}
