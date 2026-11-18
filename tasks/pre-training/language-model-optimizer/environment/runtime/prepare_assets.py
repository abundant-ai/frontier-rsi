"""Build-time only. Create disjoint public and private offline assets."""
import argparse,hashlib,json,os
from pathlib import Path
from urllib.parse import urlparse
import numpy as np


def digest(s):return hashlib.sha256(s.encode()).hexdigest()


def tokenizer_signature(tokenizer):
 # Compare actual tokenization, not chat templates: base/instruct checkpoints
 # deliberately use different prompting conventions. Evaluation uses the fixed
 # policy tokenizer, so every tensor row must retain the same token meaning.
 backend=json.loads(tokenizer.backend_tokenizer.to_str())
 backend.pop('padding',None);backend.pop('truncation',None)
 return {'vocabulary':tokenizer.get_vocab(),'backend':backend}


def main():
 from datasets import load_dataset
 from transformers import AutoTokenizer
 from huggingface_hub import snapshot_download
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['text','post','kernel','merge','service'],required=True);p.add_argument('--root',default='/opt/assets');a=p.parse_args()
 root=Path(a.root);public=root/'public';private=root/'private'
 public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
 lock=json.loads(Path('/opt/runtime/assets.lock.json').read_text())
 manifest={'upstreams':lock,'files':{}}
 if a.family=='service':
  cfg=lock['policy'];snapshot_download(cfg['repo'],revision=cfg['revision'],local_dir=public/'model',ignore_patterns=['*.png','*.jpg','*.gguf','original/*','README.md'])
 elif a.family=='text':
  cfg=lock['tokenizer'];tok=AutoTokenizer.from_pretrained(cfg['repo'],revision=cfg['revision'])
  tok.save_pretrained(public/'tokenizer')
  cfg=lock['fineweb'];ds=load_dataset(cfg['repo'],name='sample-10BT',split='train',revision=cfg['revision'],streaming=True)
  sizes={'train':100_000_000,'dev':1_000_000,'eval':2_000_000}
  buffers={k:[] for k in sizes};lengths={k:0 for k in sizes};docs=[];buckets=[];scores=[];seen=set()
  for record in ds:
   text=record['text'];key=digest(text)
   if key in seen:continue
   seen.add(key);split={'00':'eval','01':'eval','02':'dev'}.get(key[:2],'train')
   if lengths[split]>=sizes[split]:continue
   ids=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
   ids=ids[:sizes[split]-lengths[split]]
   if not ids:continue
   if split=='train' and len(ids)>=2:
    docs.append((lengths[split],len(ids)))
    host=urlparse(record.get('url','')).hostname or ''
    buckets.append(int(digest(host)[:8],16)%8)
    scores.append(len(set(ids))/len(ids))
   buffers[split].append(np.asarray(ids,dtype=np.int32));lengths[split]+=len(ids)
   if all(lengths[k]>=sizes[k] for k in sizes):break
  if any(lengths[k]<sizes[k] for k in sizes):raise RuntimeError('Insufficient disjoint corpus; refusing a partial asset bundle.')
  for split,parts in buffers.items():
   dest=private if split=='eval' else public
   np.save(dest/f'{split}.npy',np.concatenate(parts))
  np.save(public/'documents.npy',np.asarray(docs,dtype=np.int64))
  packing_docs=[(offset,min(n,2048)) for offset,n in docs if n>=64][:10000]
  if len(packing_docs)!=10000:raise RuntimeError('Insufficient packing documents of at least 64 tokens')
  np.save(public/'packing_documents.npy',np.asarray(packing_docs,dtype=np.int64))
  np.save(public/'buckets.npy',np.asarray(buckets,dtype=np.int64))
  np.save(public/'metadata.npy',np.asarray(scores,dtype=np.float32))
 elif a.family in ('post','merge'):
  cfg=lock['policy'];snapshot_download(cfg['repo'],revision=cfg['revision'],local_dir=public/'model',ignore_patterns=['*.png','*.jpg','*.gguf','original/*','README.md'])
  cfg=lock['math'];ds=load_dataset(cfg['repo'],'main',revision=cfg['revision'])
  train=list(ds['train']);dev=train[5000:5500];train=train[:5000];final=list(ds['test'])[:500]
  for name,rows,dest in [('math_train',train,public),('math_dev',dev,public),('math_eval',final,private)]:
   (dest/f'{name}.json').write_text(json.dumps(rows))
  cfg=lock['preferences'];ds=load_dataset(cfg['repo'],revision=cfg['revision'])
  # Hash whole records so identical examples cannot appear in different splits.
  rows={digest(json.dumps(r,sort_keys=True)):r for split in ds.values() for r in split}
  rows=[v for _,v in sorted(rows.items())]
  policy_tokenizer=AutoTokenizer.from_pretrained(public/'model',local_files_only=True)
  filtered=[]
  for row in rows:
   if row['chosen']==row['rejected']:continue
   if all(len(policy_tokenizer.apply_chat_template(row[k],tokenize=True))<=2048 for k in ('chosen','rejected')):filtered.append(row)
   if len(filtered)>=12500:break
  rows=filtered
  if len(rows)<12500:raise RuntimeError('Insufficient preference pairs')
  for name,items,dest in [('pref_train',rows[:10000],public),('pref_dev',rows[10000:10500],public),('pref_eval',rows[10500:11500],private)]:
   (dest/f'{name}.json').write_text(json.dumps(items))
  if a.family=='merge':
   architectures=[];tokenizations=[]
   architecture_keys=('model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','tie_word_embeddings')
   canonical=tokenizer_signature(policy_tokenizer)
   for i in range(4):
    cfg=lock['merge_'+str(i)];dest=public/'checkpoints'/str(i)
    snapshot_download(cfg['repo'],revision=cfg['revision'],local_dir=dest,allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
    configuration=json.loads((dest/'config.json').read_text());architectures.append({k:configuration.get(k) for k in architecture_keys})
    tokenizer=AutoTokenizer.from_pretrained(dest,local_files_only=True);tokenizations.append(tokenizer_signature(tokenizer))
   if any(a!=architectures[0] for a in architectures) or any(t!=canonical for t in tokenizations):raise RuntimeError('Construction sources differ in architecture or canonical tokenization')
   math_records=json.loads((public/'math_train.json').read_text())[:1000];preference_records=json.loads((public/'pref_train.json').read_text())[:1000]
   calibration=[{'prompt':r['question'],'response':r['answer']} for r in math_records]+[{'prompt':r['chosen'][0]['content'],'response':r['chosen'][-1]['content']} for r in preference_records]
   (public/'construction_calibration.json').write_text(json.dumps(calibration))
 elif a.family!='kernel':raise ValueError(a.family)
 for directory in (public,private):
  for f in directory.rglob('*'):
   if f.is_file():
    h=hashlib.sha256()
    with f.open('rb') as source:
     for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    manifest['files'][str(f.relative_to(root))]=h.hexdigest()
 (private/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 os.chmod(private,0o700)
if __name__=='__main__':main()
