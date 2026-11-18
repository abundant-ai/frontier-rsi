"""Private labels and canonical rewards for the 22 expanded research tasks."""
import json,math,statistics,sys,hashlib,gzip,os
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from sequence_worker import retrieval
from train import answer
from evaluate import score,checked_state
from model import GPT,Config

ARCH=(7,8,10,11,12,13,14,15,21)
LEARN=(20,22,27,29)


def quality_gate(passed,reason,quick,warnings):
 if passed:return
 if not quick:raise ValueError(reason)
 # A few evaluation items after 90 seconds cannot establish the full
 # task's quality threshold. Preserve the observation without presenting
 # the engineering diagnostic as a full quality result.
 warnings.append(reason)

def metric_read(path):
 value=json.loads((path/'measurement.json').read_text())
 def check(x):
  if isinstance(x,float) and not math.isfinite(x):raise ValueError('nonfinite measurement')
  if isinstance(x,dict):
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
 check(value);return value


def numerical_loss(logits,targets):
 if logits.shape[:-1]!=targets.shape or not torch.isfinite(logits).all():raise ValueError('invalid prediction tensor')
 return float(torch.nn.functional.cross_entropy(logits.flatten(0,1).float(),targets.flatten()))


def document_question(tokens,tokenizer,seed,length,batch):
 rng=np.random.default_rng(seed);identifier=int(rng.integers(10000,99999));value=str(int(rng.integers(1000,9999)))
 fact=tokenizer.encode(f"\nA verified record states that the access code for asset {identifier} is {value}.\n",add_special_tokens=False)
 prompt=tokenizer.apply_chat_template([{'role':'user','content':f"Read this document and answer the question.\n<<BODY>>\nWhat is the access code for asset {identifier}? Return only the code."}],tokenize=False,add_generation_prompt=True)
 left,right=prompt.split('<<BODY>>');prefix=tokenizer.encode(left,add_special_tokens=False);suffix=tokenizer.encode(right,add_special_tokens=False)
 n=length-len(prefix)-len(suffix)-len(fact)
 if n<1:raise ValueError('document request too short')
 offset=int(rng.integers(0,max(1,len(tokens)-n)));body=list(map(int,tokens[offset:offset+n]));position=int(rng.integers(0,len(body)+1));body[position:position]=fact
 x=torch.tensor([prefix+body+suffix],dtype=torch.long).repeat(batch,1)
 question=torch.tensor([tokenizer.encode(f"What is the access code for asset {identifier}?",add_special_tokens=False)],dtype=torch.long)
 return x,value,question


def generated_answer(prediction,tokenizer):
 ids=prediction['generated'][0].tolist()
 if tokenizer.eos_token_id in ids:ids=ids[:ids.index(tokenizer.eos_token_id)]
 return answer(tokenizer.decode(ids,skip_special_tokens=True))


def grade_seed(c,submission,out,seed,smoke,device,execute,quick=False):
 row=c['source_row'];assets=Path('/opt/assets/public');private=Path('/opt/assets/private')
 if quick and (smoke or row not in (*ARCH,*LEARN,16,18,30,31,32,33,34,48,55)):raise ValueError('quick GPU validation supports only full-size supported research tasks')
 training_seconds=90 if quick else 1800
 quality_warnings=[]
 flags=['--smoke'] if smoke else []
 def call(worker,dest,extra,timeout=1800,candidate=submission):
  command=[sys.executable,'/opt/runtime/'+worker,'--candidate',str(candidate),'--row',str(row),'--seed',str(seed),'--device',device,'--out',str(dest),*extra,*flags]
  execute(command,dest,timeout,memory_gib=None if smoke else c['memory_gib'])
  return metric_read(dest)
 def tensor_input(dest,tensors):
  dest.mkdir(parents=True,exist_ok=True);path=dest/'input.safetensors';save_file({k:v.detach().contiguous().clone() for k,v in tensors.items()},str(path));return path
 def corpus(inference=False):
  if smoke:return np.random.default_rng(seed+719).integers(4,64,65536,dtype=np.int32)
  return np.load(private/('inference_tokens.npy' if inference else 'eval.npy'),mmap_mode='r')
 if row in ARCH:
  trained=out/'trained';call('sequence_worker.py',trained,['--seconds',str(2 if smoke else training_seconds)],60 if smoke else training_seconds+10 if quick else 1800)
  needs_quality=row in (7,11,13,15);baseline=out/'quality-reference'
  if needs_quality:call('sequence_worker.py',baseline,['--seconds',str(2 if smoke else training_seconds)],60 if smoke else training_seconds+10 if quick else 1800,Path('/opt/runtime/sequence_baseline.py'))
  lengths=[16] if smoke else [8192] if row==7 else [1024] if row in (8,10,11,21) else [1024,2048,4096,8192,16384] if row==12 else [4096,8192,16384,32768] if row==14 else [2048,4096,8192,16384]
  if quick:lengths=[lengths[-1]] if row in (7,15) else sorted(set((lengths[0],lengths[-1])))
  retrieval_lengths=len(lengths)
  if not smoke and not quick:lengths=lengths*16
  losses=[];quality=[];correct=[];reference_correct=[];metrics=[];tokens=corpus()
  for index,length in enumerate(lengths):
   batch=8 if row==11 and not smoke else 1
   z=torch.tensor(np.resize(np.array(tokens[(index*length)%max(1,len(tokens)-batch*(length+1)):(index*length)%max(1,len(tokens)-batch*(length+1))+batch*(length+1)]),batch*(length+1))).long().reshape(batch,length+1)
   data=tensor_input(out/f'input-{index}',{'input':z[:,:-1].contiguous()})
   dest=out/f'predict-{index}';m=call('sequence_worker.py',dest,['--weights',str(trained),'--input',str(data),'--output-length',str(256 if row==11 else 16)],180 if smoke else 1800)
   logits=load_file(str(dest/'predictions.safetensors'))['logits'][:,-1:];losses.append(numerical_loss(logits,z[:,-1:]));metrics.append(m)
   if not smoke:
    cap=350_000_000 if row==14 else 600_000_000 if row==21 else 300_000_000 if row in (8,12) else None
    if cap and m['flops_per_token']>cap:
     raise ValueError(f"complete forward arithmetic budget exceeded: row={row}, seed={seed}, observed={m['flops_per_token']}, cap={cap}")
    if row in (12,14) and m['state_bytes']/batch>16*1024**2:raise ValueError('persistent state exceeds 16 MiB/sequence')
    if row==7 and m['decode_latency']>.010:raise ValueError(f"decoding exceeds 10 ms/token: observed={m['decode_latency']}, length={length}")
   if needs_quality:
    dest=out/f'quality-{index}';call('sequence_worker.py',dest,['--weights',str(baseline),'--input',str(data)],180 if smoke else 1800,Path('/opt/runtime/sequence_baseline.py'))
    quality.append(numerical_loss(load_file(str(dest/'predictions.safetensors'))['logits'][:,-1:],z[:,-1:]))
   if row in (13,14,15) and index<retrieval_lengths:
    # Labels remain here; only query/context tokens enter the numerical worker.
    trials=1 if quick and row==15 else 2 if smoke or quick else 32
    hits=[];refhits=[]
    for trial in range(trials):
     x,label=retrieval(seed+index*1000+trial,length,64 if smoke else 50257)
     data=tensor_input(out/f'retrieval-input-{index}-{trial}',{'input':x})
     dest=out/f'retrieval-{index}-{trial}';call('sequence_worker.py',dest,['--weights',str(trained),'--input',str(data)],180 if smoke else 1800)
     result=load_file(str(dest/'predictions.safetensors'))['logits'];hits.append(int(result[0,-1].argmax())==label)
     if needs_quality:
      dest=out/f'retrieval-reference-{index}-{trial}';call('sequence_worker.py',dest,['--weights',str(baseline),'--input',str(data)],180 if smoke else 1800,Path('/opt/runtime/sequence_baseline.py'))
      refhits.append(int(load_file(str(dest/'predictions.safetensors'))['logits'][0,-1].argmax())==label)
    correct.extend(hits);reference_correct.extend(refhits)
  if row==15:
   z=torch.tensor(np.array(tokens[:17 if smoke else 1025])).long()[None,:];data=tensor_input(out/'short-input',{'input':z[:,:-1].contiguous()});short=[]
   for name,weights,source in [('candidate',trained,submission),('reference',baseline,Path('/opt/runtime/sequence_baseline.py'))]:
    dest=out/f'short-{name}';call('sequence_worker.py',dest,['--weights',str(weights),'--input',str(data)],180 if smoke else 1800,source)
    short.append(numerical_loss(load_file(str(dest/'predictions.safetensors'))['logits'][:,-1:],z[:,-1:]))
   if not smoke:quality_gate(short[0]<=short[1]+.02,'short-context loss regression exceeds .02',quick,quality_warnings)
  if not smoke and quality:quality_gate(statistics.mean(losses)<=statistics.mean(quality)+.05,'quality-reference loss regression exceeds .05',quick,quality_warnings)
  if not smoke and reference_correct:quality_gate(statistics.mean(correct)>=statistics.mean(reference_correct)-.02,'retrieval accuracy regression exceeds 2 percentage points',quick,quality_warnings)
  value=statistics.mean(m['state_bytes']/m['batch']/m['length'] for m in metrics) if row==7 else (8*256/statistics.mean(m['decode_latency']*256 for m in metrics)) if row==11 else statistics.mean(m['latency'] for m in metrics) if row==13 else statistics.mean(correct) if row in (14,15) else statistics.mean(losses)
  return value,{'losses':losses,'quality_reference_losses':quality,'measurements':metrics,'retrieval_accuracy':statistics.mean(correct) if correct else None,'retrieval_hits':sum(correct),'retrieval_cases':len(correct),**({'diagnostic_quality_warnings':quality_warnings} if quick else {})}
 if row in LEARN:
  trained=out/'trained';measurement=call('learning_worker.py',trained,['--seconds',str(2 if smoke else training_seconds)],60 if smoke else training_seconds+10 if quick else 1800)
  if row==22:
   records=[{'question':'2+2?','answer':'4'}]*2 if smoke else json.loads((private/'research_math.json').read_text())
   if quick:records=records[:3]
   path=out/'questions.json';path.write_text(json.dumps([r['question'] for r in records]));dest=out/'answers'
   result=call('learning_worker.py',dest,['--weights',str(trained),'--input',str(path)],180 if smoke else 16000)
   if len(result['answers'])!=len(records):raise ValueError('missing final answer')
   if len(result['generated_tokens'])!=len(records) or any(n<0 or n>8192 for n in result['generated_tokens']):raise ValueError('answer token allowance')
   measurement.update(mean_generated_tokens=statistics.mean(result['generated_tokens']),missing_numeric_answers=sum(x is None for x in result['answers']))
   value=statistics.mean(str(x)==str(answer(r['answer'])) for x,r in zip(result['answers'],records))
  elif row in (27,29):
   cfg=Config(vocab=64,width=32,layers=2,heads=4,context=64) if smoke else Config();student=GPT(cfg);checked_state(student,trained/'model.safetensors');student.to(device).eval();tokens=corpus();hits=0;count=0;length=16 if smoke else 1024
   with torch.no_grad():
    for index in range(2 if smoke else min(3 if quick else 1953,len(tokens)//(length+1))):
     z=torch.tensor(np.array(tokens[index*length:index*length+length+1]),device=device).long()[None,:]
     logits=student(z[:,:-1]);hits+=int((logits.argmax(-1)==z[:,1:]).sum());count+=length
   value=hits/count
  else:
   if smoke:
    from inference_core import frozen
    expected,_=frozen(assets,'cpu',True);expected.load_state_dict(load_file(str(trained/'model.safetensors')),strict=True);value=0.
   else:
    weights=trained/'model.safetensors'
    scoring={'limit':3,'max_new_tokens':512} if quick else {}
    reasoning=score({'family':'post','mode':'policy'},weights,private/'math_eval.json',assets,device,**scoring)
    preference=score({'family':'post','mode':'adapt'},weights,private/'pref_eval.json',assets,device,**scoring)
    value=(reasoning+preference)/2
  return value,measurement
 # Frozen-model quality/performance and request-serving tasks.
 tokens=corpus(True);tokenizer=None
 if not smoke and row in (32,33,34):
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained(assets/'model',local_files_only=True)
 measurements=[];losses=[];base_losses=[];accuracies=[];base_accuracies=[];agreement=[];retained=[]
 calibration=torch.tensor(np.random.default_rng(18).integers(4,64,(2,16))).long() if smoke else torch.tensor(np.load(assets/'calibration.npy')).long()
 calibration_path=tensor_input(out/'calibration',{'input':calibration})
 if row in (18,):
  records=[{'question':'2+2?','answer':'4'}]*2 if smoke else json.loads((private/'research_math.json').read_text())
  if quick:records=records[:3]
  path=out/'questions.json';path.write_text(json.dumps([r['question'] for r in records]));dest=out/'answers';result=call('inference_worker.py',dest,['--input',str(path)],180 if smoke else 16000)
  if len(result['answers'])!=len(records) or any(n>8192 for n in result['generated_tokens']):raise ValueError('answer count or token allowance')
  return statistics.mean(str(x)==str(answer(r['answer'])) for x,r in zip(result['answers'],records)),result
 if row in (48,55):
  traces=[]
  for rate in ([1.] if smoke else [.5] if quick else [.5,1.,2.,4.]):
   requests=[]
   for i in range(2 if smoke else 4 if quick else 2000):
    length=8 if smoke else [128,512,2048,8192][i%4];offset=i*97%max(1,len(tokens)-length)
    requests.append({'input':list(map(int,tokens[offset:offset+length])),'output_length':2 if smoke else 32 if quick else [32,64,128,512][i%4]})
   path=out/f'trace-{rate}.json';path.write_text(json.dumps(requests));results=[]
   replay_path=out/f'trace-{rate}-execution-plan.json'
   for phase in ('candidate','reference'):
    extra=['--baseline','--replay-plan',str(replay_path)] if phase=='reference' else []
    dest=out/f'trace-{rate}-{phase}';measurement=call('inference_worker.py',dest,['--input',str(path),'--rate',str(rate)]+extra,180 if smoke else 16000)
    if phase=='candidate':
     plan=measurement.pop('execution_plan',None)
     if not isinstance(plan,list) or not plan:raise ValueError('missing trusted serving execution plan')
     if not math.isclose(float(plan[-1]['finished_at']),measurement['elapsed'],abs_tol=1e-9):raise ValueError('serving plan elapsed time differs')
     encoded=json.dumps(plan,separators=(',',':'),allow_nan=False).encode();replay_path.write_bytes(encoded);replay_path.chmod(0o444)
     archive=Path('/logs/verifier/serving-plans') if os.geteuid()==0 else out/'serving-plans'
     archive.mkdir(parents=True,exist_ok=True)
     (archive/f'row-{row}-seed-{seed}-rate-{rate}.json.gz').write_bytes(gzip.compress(encoded,mtime=0))
     measurement.update(execution_plan_sha256=hashlib.sha256(encoded).hexdigest(),execution_plan_steps=len(plan))
    results.append(measurement)
   result,baseline=results
   if result['outputs']!=baseline['outputs']:raise ValueError('serving changed frozen-model outputs')
   if result['requests']!=len(requests):raise ValueError('request dropped')
   if not smoke and (result['ttft_p95']>2 or result['intertoken_p95']>.05):raise ValueError(f"serving tail-latency limit: row={row}, ttft_p95={result['ttft_p95']}, intertoken_p95={result['intertoken_p95']}")
   traces.append(result)
  return statistics.mean(r['output_tokens' if row==48 else 'requests']/r['elapsed'] for r in traces),{'traces':traces}
 cases=2 if smoke else 2000 if row in (16,30) else 1000 if row in (32,33) else 8
 # The float-cache no-op fits the real 4096-token case. Longer full-task
 # cases deliberately require the excluded quantization technique; those
 # remain in full grading, not in this reduced engineering diagnostic.
 indices=([0] if row==34 else [0,3]) if quick else range(cases)
 for index in indices:
  length=16 if smoke else [128,512,1024,4096][index%4] if row in (16,30,31) else [4096,8192,16384,32768][index%4] if row==32 else [4096,8192,16384][index%3]
  batch=1 if smoke or row in (32,33) or index%2==0 else 8
  offset=index*97%max(1,len(tokens)-batch*(length+1));z=torch.tensor(np.resize(np.array(tokens[offset:offset+batch*(length+1)]),batch*(length+1))).long().reshape(batch,length+1)
  label=None
  question=z[:,-3:-1].clone()
  retrieval_case=row in (32,33) or row==34 and index%2==1
  if retrieval_case:
   x,label=retrieval(seed+index,length,64 if smoke else 151936);x=x.repeat(batch,1);z=torch.cat((x,torch.full((batch,1),label,dtype=torch.long)),-1)
   if not smoke:
    x,label,question=document_question(tokens,tokenizer,seed+index,length,batch);target=tokenizer.encode(label,add_special_tokens=False)[0];z=torch.cat((x,torch.full((batch,1),target,dtype=torch.long)),-1)
  path=tensor_input(out/f'input-{index}',{'input':z[:,:-1].contiguous(),'question':question})
  results=[];predictions=[]
  for phase in ('candidate','reference'):
   dest=out/f'case-{index}-{phase}'
   m=call('inference_worker.py',dest,['--input',str(path),'--calibration',str(calibration_path),*(['--prepared',str(out/'case-0-candidate'/'representation')] if row==31 and phase=='candidate' else []),'--output-length',str(2 if smoke else 16 if quick else ([32,64,128,256] if row==16 else [64,128,256,512])[index%4])]+(['--baseline'] if phase=='reference' else []),180 if smoke else 1800)
   results.append(m);predictions.append(load_file(str(dest/'predictions.safetensors')))
  m,baseline=results;pred,truth=predictions;measurements.append(m)
  if not smoke:
   if row==31 and m['payload_bytes']>1.5*1024**3:raise ValueError('model payload exceeds 1.5 GiB')
   if row in (32,34) and m['state_bytes']/batch>(64 if row==32 else 128)*1024**2:raise ValueError(f"persistent cache cap: row={row}, observed_bytes_per_sequence={m['state_bytes']/batch}, limit_mib={64 if row==32 else 128}")
   if row==32 and m['decode_latency']>.020:raise ValueError(f"decode exceeds 20 ms/token: observed={m['decode_latency']}, length={length}")
   if row==33 and m['preprocess_latency']>2:raise ValueError('preprocessing exceeds 2 seconds')
  if row==33:
   retained.append(m['input_tokens']);accuracies.append((generated_answer(pred,tokenizer)==label if not smoke else int(pred['logits'][0,-1].argmax())==label));base_accuracies.append((generated_answer(truth,tokenizer)==label if not smoke else int(truth['logits'][0,-1].argmax())==label))
  else:
   losses.append(numerical_loss(pred['logits'],z[:,-1:]));base_losses.append(numerical_loss(truth['logits'],z[:,-1:]))
   accuracies.append(float((pred['logits'].argmax(-1)==z[:,-1:]).float().mean()) if label is None else float((generated_answer(pred,tokenizer)==label if not smoke else int(pred['logits'][0,-1].argmax())==label)))
   base_accuracies.append(float((truth['logits'].argmax(-1)==z[:,-1:]).float().mean()) if label is None else float((generated_answer(truth,tokenizer)==label if not smoke else int(truth['logits'][0,-1].argmax())==label)))
  if row==30:
   agreement.append(float((pred['generated']==truth['generated']).float().mean()))
   # Frozen target probabilities are checked as well as greedy token identity.
   if not torch.allclose(pred['logits'].float().softmax(-1),truth['logits'].float().softmax(-1),atol=1e-5,rtol=1e-3):raise ValueError('target distribution differs')
 if not smoke:
  tolerance=.02 if row==16 else .05
  if row in (16,31,34) and statistics.mean(losses)>statistics.mean(base_losses)+tolerance:
   quality_gate(False,f'frozen-model loss regression: row={row}, seed={seed}, candidate_mean={statistics.mean(losses)}, reference_mean={statistics.mean(base_losses)}, delta={statistics.mean(losses)-statistics.mean(base_losses)}, tolerance={tolerance}',quick,quality_warnings)
  if row in (16,33,34):quality_gate(statistics.mean(accuracies)>=statistics.mean(base_accuracies)-(.01 if row==16 else .02),'frozen-model accuracy regression',quick,quality_warnings)
  if agreement and statistics.mean(agreement)<.999:raise ValueError('greedy agreement below 99.9%')
 value=statistics.mean(accuracies) if row==32 else statistics.mean(retained) if row==33 else statistics.mean((2 if smoke else 16 if quick else [64,128,256,512][i%4])*m['batch']/m['elapsed'] for i,m in enumerate(measurements)) if row==30 else statistics.mean(m['elapsed'] for m in measurements) if row==16 else statistics.mean(m['decode_latency'] for m in measurements)
 return value,{'measurements':measurements,'losses':losses,'reference_losses':base_losses,'accuracy':statistics.mean(accuracies),**({'diagnostic_quality_warnings':quality_warnings} if quick else {})}
