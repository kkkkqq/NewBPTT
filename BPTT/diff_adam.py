from BPTT.diff_optim import DiffOptimizer
from type_ import *
import torch
from functools import partial
class DiffAdam(DiffOptimizer):

    def __init__(self,
                 model:Module, 
                 optimizer:Optimizer, 
                 tape_on_device:bool=True,
                 _attpa2modpa_idxes:List[int]=None)->None:
        super().__init__(model, optimizer, tape_on_device, _attpa2modpa_idxes)
        self.state_vars.update({'dLdm_groups':[],
                                'dLdv_groups':[],
                                'v_groups':[],
                                't_groups':[]})
        self.states_tape_dct = {'m_groups':[], 'v_groups':[], 't_groups':[]}
        self.coefs_tape_dct = {'lr_groups':[],
                               'beta1_groups':[],
                               'beta2_groups':[],
                               'eps_groups':[],
                               'maximize_groups':[],
                               'weight_decay_groups':[],
                               }
        for group in optimizer.param_groups:
            if group['amsgrad']:
                raise NotImplementedError('havenot implement amsgrad for adam!')

    def update_backprop_state(self, 
                              train_lr:bool,
                              params:List[List[Tensor]],
                              grads:List[List[Tensor]],
                              group_startends:List[Tuple[int,int]],
                              dLdw_groups:List[Tensor],
                              dLdgrad_groups:List[Tensor],
                              dLdm_groups:List[Tensor],
                              dLdv_groups:List[Tensor],
                              m_groups:List[Tensor],
                              v_groups:List[Tensor],
                              lr_groups:List[float],
                              beta1_groups:List[float],
                              beta2_groups:List[float],
                              eps_groups:List[float],
                              weight_decay_groups:List[float],
                              maximize_groups:List[bool],
                              t_groups:List[Tensor]
                              ):
        from torch import zeros_like, no_grad, cat, sqrt, tensor, ones
        with no_grad():
            if len(dLdm_groups)==0:
                for dLdw_ in dLdw_groups:
                    dLdm_groups.append(zeros_like(dLdw_))
                    dLdv_groups.append(zeros_like(dLdw_))
            # for idx, group in enumerate(param_groups):
            dLdgrad_groups.clear()
            groups = zip(group_startends,
                         grads,
                         dLdw_groups, 
                         dLdm_groups, 
                         dLdv_groups, 
                         t_groups,
                         m_groups, 
                         v_groups,
                         lr_groups,
                         beta1_groups,
                         beta2_groups,
                         eps_groups,
                         weight_decay_groups,
                         maximize_groups,
                      )
            for (start,end), grad, dLdw, dLdm, dLdv, t, m, v, lr, beta1, beta2, eps, weight_decay, maximize in groups:
                if m.device != dLdw.device:
                    device_ = dLdw.device
                    m = m.to(device_)
                    v = v.to(device_)
                gt = grad.detach().clone()
                beta1, beta2 = tensor(beta1), tensor(beta2)
                omb1 = 1.-beta1
                omb1t = 1.- beta1.pow(t)
                #omb1t = 1. - beta1**t
                omb2 = 1.-beta2
                omb2t = 1. - beta2.pow(t)
                # omb2t = 1. - beta2**t
                rtvdomb2t = sqrt(v.div(omb2t))
                rtvdomb2taddeps = rtvdomb2t.add(eps)
                if train_lr:
                    a = m.div(rtvdomb2taddeps).div(omb1t)
                    self.lr_grad.add_(dLdw.dot(a).mul(-1))
                if maximize:
                    gt.mul_(-1.)
                if weight_decay!=0:
                    w = cat([ele.flatten() for ele in params[start:end]])
                    gt.add_(w.mul(weight_decay))
                dLdm.mul_(beta1).sub_(dLdw.mul(lr).div(omb1t).div(rtvdomb2taddeps))
                dLdv.mul_(beta2).add_(dLdw.mul(lr).mul(m).div(omb1t).div(omb2t).div(2).div(rtvdomb2t).div(rtvdomb2taddeps.pow(2)))
                # dLdv.mul_(beta2).add_(ones(1, device=device_).div(omb1t).div(omb2t).div(2).mul(lr).mul(dLdw).mul(m).div(rtvdomb2t).div(rtvdomb2taddeps.pow(2)))
                gt.mul_(2).mul_(omb2).mul_(dLdv).add_(dLdm.mul(omb1))
                if weight_decay!=0:
                    dLdw.add_(gt.mul(weight_decay))
                if maximize:
                    gt.mul_(-1.)
                dLdgrad_groups.append(gt)
                # print('max dLdw', torch.max(torch.abs(dLdw)).item())
                # print('max dLdgt', torch.max(torch.abs(gt)).item())
        return None
    
    def _pre_step(self,
                  taped,
                  tape_on_device,
                  model_tape):
        from torch import no_grad, randn_like
        super()._pre_step(taped, tape_on_device, model_tape)
        if self.cur_idx == 0:
            with no_grad():
                for pa in self.attached_params:
                    pa.grad.add_(randn_like(pa)*1e-10)

    def _post_step(self, 
                   taped,
                   tape_on_device,
                   state_tapes_dct,
                   coef_tapes_dct):
        from torch import cat
        if taped:
            state = self.opt_state
            param_groups = self.opt_param_groups
            groups_tup = [[] for _ in range(9)]
            state_keys = ['m_groups', 'v_groups', 't_groups']
            coef_keys = ['lr_groups', 'beta1_groups', 'beta2_groups', 'eps_groups', 'maximize_groups', 'weight_decay_groups']
            for group in param_groups:
                pas = group['params']
                m_lst = []
                v_lst = []
                t_lst = []
                for pa in pas:
                    dct = state[pa]
                    m_lst.append(dct['exp_avg'].flatten())
                    v_lst.append(dct['exp_avg_sq'].flatten())
                    t_lst.append(dct['step'].item())
                    
                m = cat(m_lst, dim=0)
                v = cat(v_lst, dim=0)
                t_set = set(t_lst)
                if len(t_set)>1:
                    raise AssertionError("'step' in optimizer differ for each params in the param group!")
                else:
                    t = t_set.pop()
                if not tape_on_device:
                    m = m.to('cpu')
                    v = v.to('cpu')
                lr = group['lr']
                betas = group['betas']
                beta1, beta2 = betas
                eps = group['eps']
                maximize=group['maximize']
                weight_decay=group['weight_decay']
                items = (m,v,t,lr,beta1,beta2,eps,maximize,weight_decay)
                for groups, item in zip(groups_tup, items):
                    groups.append(item)
                state_groups = groups_tup[:3]
                coef_groups = groups_tup[3:]
            states = {k:v for k,v in zip(state_keys, state_groups)}
            coefs = {k:v for k,v in zip(coef_keys, coef_groups)}
            for stk in state_keys:
                state_tapes_dct[stk].append(states[stk])
            for ck in coef_keys:
                coef_tapes_dct[ck].append(coefs[ck])
        else:
            for val in state_tapes_dct.values():
                val.append(None)
            for val in coef_tapes_dct.values():
                val.append(None)
        return None


                


