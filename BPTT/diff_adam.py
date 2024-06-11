from BPTT.diff_optim import DiffOptimizer
from type_ import *
import torch

class Adam(DiffOptimizer):

    def __init__(self,
                 model:Module, 
                 optimizer:Optimizer, 
                 tape_on_device:bool=True,
                 paidx_grpidx_map:List[List[Tuple[int,int]]]=None)->None:
        super().__init__(model, optimizer, True, tape_on_device, paidx_grpidx_map)
        self.dLdv_groups:List[Tensor] = None
        self.dLdm_groups:List[Tensor] = None

    def update_backprop_state(self):
        from torch import zeros_like, no_grad, cat
        from numpy import power
        dLdw_groups = self.dLdw_groups
        if self.dLdv_groups is None:
            dLdv_groups:List[Tensor] = []
            self.dLdv_groups = dLdv_groups
            for dLdw in dLdw_groups:
                dLdv_groups.append(zeros_like(dLdw))
        if self.dLdm_groups is None:
            dLdm_groups:List[Tensor] = []
            self.dLdm_groups = dLdm_groups
            for dLdw in dLdw_groups:
                dLdm_groups.append(zeros_like(dLdw))
        dLdv_groups = self.dLdv_groups
        dLdm_groups = self.dLdm_groups
        dLdgrad_groups = []
        self.dLdgrad_groups = dLdgrad_groups
        states = self.states_tape[self.cur_idx]
        param_groups = self.optimizer.param_groups
        grads_ = self._model_grads_after_backward
        gts = [grads_[startend[0]:startend[1]] for startend in self._pa_groups_startend_lst]
        with no_grad():
            # for idx, group in enumerate(param_groups):
            for dLdw, dLdm, dLdv, state, group, gt in zip(dLdw_groups, dLdm_groups, dLdv_groups, states, param_groups, gts):
                # gt = self.flatten([ele.grad.detach() for ele in group['params']]).detach()
                gt = cat([ele.detach().flatten() for ele in gt])
                lr = group['lr']
                beta1 = group['betas'][0]
                beta2 = group['betas'][1]
                eps = group['eps']
                weight_decay = group['weight_decay']
                maximize = group['maximize']
                m = cat([dct['exp_avg'].flatten() for dct in state])
                v = cat([dct['exp_avg_sq'].flatten() for dct in state])
                # m[m==0] = 1e-10#如果回头还是爆炸，就把adam的step改一下，让它在第一个step处加一个正态offset
                # v[v==0] = 1e-20
                #m.add_(1e-8)# offset m and v by a very small amount, keep dLdv from exploding
                #v.add_(1e-16)
                t = state[0]['step'].item()
                omb1 = 1.-beta1
                omb1t = 1.- power(beta1, t)
                omb2 = 1.-beta2
                omb2t = 1. - power(beta2, t)
                if maximize:
                    gt.mul_(-1.)
                if weight_decay!=0:
                    w = torch.cat([ele.detach().flatten() for ele in group['params']])
                    gt.add_(w.mul(weight_decay))
                sqrt_vdivomb2t = v.div_(omb2t).pow_(0.5)
                # print('max m', torch.max(torch.abs(m)).item())
                # print('min m', torch.min(torch.abs(m)).item())
                # print('max v', torch.max(torch.abs(v)).item())
                # print('min v', torch.min(torch.abs(v)).item())
                # print('max sqrtvdivomb2t', torch.max(torch.abs(sqrt_vdivomb2t)).item())
                # print('min sqrtvdivomb2t', torch.min(torch.abs(sqrt_vdivomb2t)).item())
                dLdm.mul_(beta1).sub_(dLdw.mul(lr/omb1t).div(sqrt_vdivomb2t.add(eps)))
                dLdv.mul_(beta2)
                # print('max dLdv before adding', torch.max(torch.abs(dLdv)).item())
                dLdv_add = m.mul_(lr/omb1t/omb2t/2.0).mul_(dLdw)
                # print('max additive before division', torch.max(torch.abs(dLdv_add)).item())
                dLdv_add.div_(sqrt_vdivomb2t)
                # print('max additive after first division', torch.max(torch.abs(dLdv_add)).item())
                dLdv_add.div_(sqrt_vdivomb2t.add_(eps).pow_(2))# this is where dLdv explodes if v and m are not slighted offset.
                # print('epsilon: ', eps)
                # print('max second division:', torch.max(torch.abs(sqrt_vdivomb2t.add(eps).pow(2))).item())
                # print('min second division:', torch.min(torch.abs(sqrt_vdivomb2t.add(eps).pow(2))).item())
                # print('max additive after division', torch.max(torch.abs(dLdv_add)).item())
                #dLdv.add_(dLdw.mul(m.mul(lr/omb1t/omb2t/2.0)).div(sqrt_vdivomb2t).div(sqrt_vdivomb2t.add(eps).pow(2)))
                dLdv.add_(dLdv_add)
                # print('max gt', torch.max(torch.abs(gt)).item())
                # print('max dLdv', torch.max(torch.abs(dLdv)).item())
                # print('max dLdm', torch.max(torch.abs(dLdm)).item())
                # print('omb2', omb2)
                # print('omb1', omb1)
                gt.mul_(2.*omb2).mul_(dLdv).add_(dLdm.mul(omb1))
                if weight_decay!=0:
                    dLdw.add_(gt.mul(weight_decay))
                if maximize:
                    gt.mul_(-1.)
                dLdgrad_groups.append(gt)
                # print('max dLdw', torch.max(torch.abs(dLdw)).item())
                # print('max dLdgt', torch.max(torch.abs(gt)).item())
        return None