import torch

def padded(documents,pad=0):
 n=max(len(d)-1 for d in documents);b=len(documents)
 x=torch.full((b,n),pad,dtype=torch.long);y=torch.full((b,n),-100,dtype=torch.long)
 positions=torch.zeros(b,n,dtype=torch.long)
 mask=torch.ones(b,1,n,n,dtype=torch.bool).tril()
 for i,d in enumerate(documents):
  size=len(d)-1;x[i,:size]=d[:-1];y[i,:size]=d[1:];positions[i,:size]=torch.arange(size)
 return x,y,mask,positions

def check(documents,result):
 if not isinstance(result,(tuple,list)) or len(result)!=4:raise ValueError('batch must contain four tensors')
 x,y,mask,pos=result
 if any(not isinstance(t,torch.Tensor) for t in result):raise ValueError('batch values must be tensors')
 if any(t.dtype!=torch.long for t in (x,y,pos)) or mask.dtype!=torch.bool:raise ValueError('batch dtypes')
 if x.ndim!=2 or x.numel()==0:raise ValueError('batch must be a nonempty matrix')
 if x.shape!=y.shape or x.shape!=pos.shape:raise ValueError('batch tensor shapes')
 if mask.shape!=(x.shape[0],1,x.shape[1],x.shape[1]):raise ValueError('attention mask shape')
 observed=torch.stack([x[y!=-100],y[y!=-100]],-1)
 expected=torch.cat([torch.stack((d[:-1],d[1:]),-1) for d in documents])
 if not torch.equal(observed.cpu(),expected.cpu()):raise ValueError('targets/order changed')
 # Check the complete dependency set and position for every scored token.
 cursor=0
 for d in documents:
  n=len(d)-1
  coords=(y!=-100).nonzero()[cursor:cursor+n];cursor+=n
  for j,(b,t) in enumerate(coords.tolist()):
   actual=mask[b,0,t].bool().nonzero().flatten().tolist()
   expected_cols=coords[:j+1,1].tolist()
   if not torch.equal(coords[:j+1,0],torch.full((j+1,),b)) or actual!=expected_cols or pos[b,t]!=j:
    raise ValueError('document dependencies or positions changed')
