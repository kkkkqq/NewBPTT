import torch
from torch import Tensor
import torch.jit
from typing import List, Optional

@torch.jit.script
def flatten(tsrs:List[Tensor]):
    return torch.cat([tsr.flatten() for tsr in tsrs])

@torch.jit.script
def flatten_detached(tsrs:List[Tensor]):
    return torch.cat([tsr.detach().flatten() for tsr in tsrs])

@torch.jit.script
def adam_bpstate(train_lr:bool, 
                 m:Tensor,
                 v:Tensor,
                 t:Tensor,
                 gt:Tensor,
                 w:Tensor,
                 dLdw:Tensor,
                 dLdm:Tensor,
                 dLdv:Tensor,
                 lr:float,
                 beta1:Tensor,
                 beta2:Tensor,
                 eps:float,
                 weight_decay:float,
                 maximize:bool
                 ):
    omb1 = 1.-beta1
    omb1t = 1.- beta1.pow(t)
    omb2 = 1.-beta2
    omb2t = 1. - beta2.pow(t)
    rtvdomb2t = v.div(omb2t).pow(0.5)
    rtvdomb2taddeps = rtvdomb2t.add(eps)
    lr_grad = torch.zeros(1, device=dLdm.device)
    if train_lr:
        lr_grad.add(dLdw.dot(m.div(rtvdomb2taddeps).div(omb1t)).mul(-1))
    if maximize:
        gt.mul_(-1)
    if weight_decay != 0:
        gt.add_(w.mul(weight_decay))
    dLdm.mul_(beta1).sub_(dLdw.mul(lr).div(omb1t).div(rtvdomb2taddeps))
    dLdv.mul_(beta2).add_(dLdw.mul(lr).mul(m).div(omb1t).div(omb2t).div(2).div(rtvdomb2t).div(rtvdomb2taddeps.pow(2)))
    gt.mul_(2).mul_(omb2).mul_(dLdv).add_(dLdm.mul(omb1))
    if weight_decay!=0:
        dLdw.add_(gt.mul(weight_decay))
    if maximize:
        gt.mul_(-1.)
    dLdgrad = gt
    return dLdgrad, lr_grad
