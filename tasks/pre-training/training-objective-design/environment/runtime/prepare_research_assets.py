"""Build-time transformations for task-specific accessible data allowances."""
import argparse,json,secrets,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file


def private_math(seed,count):
 rng=np.random.default_rng(seed);rows=[]
 for i in range(count):
  a,b,c=map(int,rng.integers(2,10000,3));kind=i%4
  question,value=[(f'A warehouse has {a} boxes with {b} bolts each and receives {c} more bolts. How many bolts are there?',a*b+c),(f'A club starts with {a*b+c} tickets, sells {c}, and divides the rest equally among {b} groups. How many tickets per group?',a),(f'A train travels {a} km each hour for {b} hours, then another {c} km. What is the total distance?',a*b+c),(f'A store orders {a} packs of {b} pens and gives away {c} pens. How many pens remain?',a*b-c)][kind]
  rows.append({'question':question,'answer':str(value)})
 return rows


def main():
 p=argparse.ArgumentParser();p.add_argument('--row',type=int,required=True);p.add_argument('--root',default='/opt/assets');p.add_argument('--diagnostic',action='store_true');a=p.parse_args();root=Path(a.root);public=root/'public';private=root/'private'
 if a.row in (18,22):
  # Private, freshly authored arithmetic instances are separate from GSM8K.
  (private/'research_math.json').write_text(json.dumps(private_math(secrets.randbits(32),500)))
 if a.row==20:
  pairs=json.loads((public/'pref_train.json').read_text())
  (public/'feedback_labels.json').write_text(json.dumps(pairs[:1000]))
  (public/'feedback_prompts.json').write_text(json.dumps([r['chosen'][0]['content'] for r in pairs[:10000]]))
  (public/'pref_train.json').unlink()
  (public/'math_train.json').unlink(missing_ok=True)
 if a.row==18:
  from datasets import load_dataset
  lock=json.loads(Path('/opt/runtime/assets.lock.json').read_text());cfg=lock['math'];data=load_dataset(cfg['repo'],'main',revision=cfg['revision'])
  (public/'math_dev.json').write_text(json.dumps(list(data['train'])[5000:5500]))
 if a.row==22:
  from huggingface_hub import hf_hub_download
  lock=json.loads(Path('/opt/runtime/assets.lock.json').read_text());cfg=lock['process_dev'];diagnostics={}
  for name in ('gsm8k','math','olympiadbench','omnimath'):
   source=hf_hub_download(cfg['repo'],name+'.json',repo_type='dataset',revision=cfg['revision'])
   diagnostics[name]=json.loads(Path(source).read_text())
  (public/'process_development.json').write_text(json.dumps(diagnostics))
  rows=json.loads((public/'math_train.json').read_text());records=[]
  for i in range(10000):
   r=rows[i%len(rows)];lines=r['answer'].splitlines();text='\n'.join(lines)
   # Each record exposes at most two annotated positions; unannotated positions
   # are masked in training. Alternating controlled corruptions provide negatives.
   if i>=len(rows):
    import re
    numbers=re.findall(r'[-+]?\d+(?:\.\d+)?',text.replace(',',''));wrong=str(float(numbers[-1])+1) if numbers else '0'
    text=text+'\nTherefore the answer is '+wrong+'.'
   positions=[0,max(0,len(text.splitlines())-1)]
   labels=[-1.]*len(text.splitlines())
   for pos in set(positions):labels[pos]=0. if i>=len(rows) and pos==positions[-1] else 1.
   records.append({'text':text,'labels':labels,'source':'gsm8k-training-derived','corrupted':i>=len(rows)})
  (public/'solution_records.json').write_text(json.dumps(records))
 if a.row==29:
  tokens=np.load(public/'train.npy',mmap_mode='r');subset=np.array(tokens[:1_000_000]);del tokens;np.save(public/'train.npy',subset)
  for name in ('documents.npy','packing_documents.npy','buckets.npy','metadata.npy'): (public/name).unlink(missing_ok=True)
 if a.row in (16,18,30,31,32,33,34,48,55):
  # GPT-2 token ids must never be interpreted as Qwen ids. Retokenize offline.
  from transformers import AutoTokenizer
  source=AutoTokenizer.from_pretrained(public/'tokenizer',local_files_only=True);target=AutoTokenizer.from_pretrained(public/'model',local_files_only=True)
  for directory,name in [(public,'dev'),(private,'eval')]:
   tokens=np.load(directory/f'{name}.npy');ids=[]
   for offset in range(0,len(tokens),16384):ids.extend(target.encode(source.decode(tokens[offset:offset+16384]),add_special_tokens=False))
   np.save(directory/'inference_tokens.npy',np.array(ids,dtype=np.int32))
  calibration=np.load(public/'train.npy',mmap_mode='r')[:400000]
  ids=target.encode(source.decode(calibration),add_special_tokens=False)
  if len(ids)<128*2048:raise ValueError('insufficient calibration tokens')
  np.save(public/'calibration.npy',np.array(ids[:128*2048],dtype=np.int32).reshape(128,2048))
 if a.row==15:
  from sequence_worker import setup
  from torch.nn import functional as F
  model,c=setup('/opt/runtime/sequence_baseline.py',15,17,'cpu',False);opt=torch.optim.AdamW(model.parameters(),lr=6e-4)
  tokens=np.load(public/'train.npy',mmap_mode='r');start=time.monotonic();rng=np.random.default_rng(17);steps=0
  pretrain_seconds=90 if a.diagnostic else 1800
  while time.monotonic()-start<pretrain_seconds:
   offset=int(rng.integers(0,len(tokens)-1025));z=torch.tensor(np.array(tokens[offset:offset+1025])).long()[None,:]
   opt.zero_grad();logits,_=model(z[:,:-1]);loss=F.cross_entropy(logits.flatten(0,1),z[:,1:].flatten());loss.backward();opt.step();steps+=1
  if not steps:raise ValueError('no short-context pretraining completed')
  save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(public/'position_start.safetensors'))
  (public/'position_start.json').write_text(json.dumps({'seed':17,'context':1024,'steps':steps,'seconds':time.monotonic()-start,'diagnostic_only':a.diagnostic}))
 # Refresh checksums after trimming/re-tokenizing assets.
 import hashlib
 manifest={'files':{}}
 for directory in (public,private):
  for f in directory.rglob('*'):
   if f.is_file():
    h=hashlib.sha256()
    with f.open('rb') as stream:
     for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    manifest['files'][str(f.relative_to(root))]=h.hexdigest()
 (private/'research-manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
