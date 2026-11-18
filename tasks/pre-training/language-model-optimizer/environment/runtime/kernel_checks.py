"""Deterministic numerical contracts for probability vectors and sample support."""
import torch


def sampling(actual,expected,a,smoke=False):
 if len(actual)!=2:raise ValueError('sampling returns probabilities and sampled indices')
 probability,samples=actual;truth=expected[0]
 if probability.shape!=truth.shape or not torch.isfinite(probability).all():raise ValueError('probability shape or finiteness')
 if (probability<0).any() or not torch.allclose(probability.sum(-1),torch.ones_like(probability.sum(-1)),atol=1e-5,rtol=1e-5):raise ValueError('invalid probability distribution')
 if float((probability.float()-truth.float()).norm()/truth.float().norm().clamp_min(1e-8))>1e-4:raise ValueError('probabilities differ')
 if not torch.equal(probability>0,truth>0):raise ValueError('sampling support differs')
 if samples.dtype not in (torch.int32,torch.int64) or samples.shape!=a['uniforms'].shape:raise ValueError('sample dtype or shape')
 if (samples<0).any() or (samples>=truth.shape[-1]).any():raise ValueError('sample index out of range')
 if (truth.gather(-1,samples.long())<=0).any():raise ValueError('sample outside permitted support')
 if truth.shape[-1]==16:
  for row in range(len(truth)):
   counts=torch.bincount(samples[row].long(),minlength=16).float()/samples.shape[-1]
   tv=float((counts-truth[row]).abs().sum()/2)
   if tv>(.08 if smoke else .01):raise ValueError('sampling total variation exceeds contract')
