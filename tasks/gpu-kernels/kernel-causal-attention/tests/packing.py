import torch

def padded(documents,pad=0):
 n=max(len(d)-1 for d in documents);b=len(documents)
 x=torch.full((b,n),pad,dtype=torch.long);y=torch.full((b,n),-100,dtype=torch.long)
 positions=torch.zeros(b,n,dtype=torch.long)
 mask=torch.ones(b,1,n,n,dtype=torch.bool).tril()
 for i,d in enumerate(documents):
  size=len(d)-1;x[i,:size]=d[:-1];y[i,:size]=d[1:];positions[i,:size]=torch.arange(size)
 return x,y,mask,positions
