"""Deliberately simple GPU implementation; no autotuning or persistent work."""
import torch


def matmul(a,b):
    if not a.is_cuda:return a@b
    import triton
    import triton.language as tl
    @triton.jit
    def operation(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,SA:tl.constexpr,SB:tl.constexpr,T:tl.constexpr):
        row=tl.program_id(0)*T+tl.arange(0,T)
        col=tl.program_id(1)*T+tl.arange(0,T)
        r=tl.arange(0,T);acc=tl.full((T,T),0,tl.float32)
        for base in range(triton.cdiv(K,T)):
            k=base*T+r
            av=tl.load(A+row[:,None]*SA+k[None,:],(row[:,None]<M)&(k[None,:]<K),0)
            bv=tl.load(B+k[:,None]*SB+col[None,:],(k[:,None]<K)&(col[None,:]<N),0)
            acc+=tl.dot(av,bv)
        tl.store(C+row[:,None]*N+col[None,:],acc,(row[:,None]<M)&(col[None,:]<N))
    out=torch.empty((a.shape[0],b.shape[1]),device=a.device,dtype=a.dtype)
    operation[(triton.cdiv(a.shape[0],16),triton.cdiv(b.shape[1],16))](a,b,out,a.shape[0],b.shape[1],a.shape[1],a.stride(0),b.stride(0),16,num_warps=4,num_stages=1)
    return out
