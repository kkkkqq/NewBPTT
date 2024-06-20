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
                              t_groups:List[Tensor],
                              tape_on_device:bool
                              ):
        from torch import zeros_like, no_grad, cat, sqrt, tensor, ones, as_tensor, zeros
        from BPTT.jits import flatten_detached, adam_bpstate
        with no_grad():
            if len(dLdm_groups)==0:
                for dLdw_ in dLdw_groups:
                    dLdm_groups.append(zeros_like(dLdw_))
                    dLdv_groups.append(zeros_like(dLdw_))
            # for idx, group in enumerate(param_groups):
            dLdgrad_groups.clear()
            # if weight decay, prepare w for later use
            w_groups = []
            for wd, (start,end) in zip(weight_decay_groups, group_startends):
                if wd!=0:
                    w_groups.append(flatten_detached(params[start:end]))
                else:
                    w_groups.append(zeros(1))
            groups = zip(w_groups,
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
            for w, grad, dLdw, dLdm, dLdv, t, m, v, lr, beta1, beta2, eps, weight_decay, maximize in groups:
                if not tape_on_device:
                    device_ = dLdw.device
                    m = m.to(device_)
                    v = v.to(device_)
                gt = grad.detach().clone()
                beta1, beta2 = as_tensor(beta1), as_tensor(beta2)
                dLdgrad, lr_grad = adam_bpstate(train_lr,
                                                m,
                                                v,
                                                t,
                                                gt,
                                                w,
                                                dLdw,
                                                dLdm,
                                                dLdv,
                                                lr,
                                                beta1,
                                                beta2,
                                                eps,
                                                weight_decay,
                                                maximize)
                self.lr_grad.add_(lr_grad)
                dLdgrad_groups.append(dLdgrad)
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
        from BPTT.jits import flatten
        if taped:
            state = self.opt_state
            param_groups = self.opt_param_groups
            groups_tup = [[] for _ in range(9)]
            state_keys = ['m_groups', 'v_groups', 't_groups']
            coef_keys = ['lr_groups', 'beta1_groups', 'beta2_groups', 'eps_groups', 'maximize_groups', 'weight_decay_groups']

            for group in param_groups:
                #首先是抽取每个pa对应的states并分别组成lst
                pas = group['params']
                m,v,t = mvt_lst(state, pas)
                m = flatten(m)
                v = flatten(v)
                t = t[0]
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


                
def mvt_lst(state, pas):
    mvt_lst_ = []
    for pa in pas:
        dct = state[pa]
        mvt_lst_.append((dct['exp_avg'], dct['exp_avg_sq'], dct['step']))
    return  zip(*mvt_lst_)
    

def m_v_t(state, pas):
    from torch import cat
    m_lst = []
    v_lst = []
    dct = dict()
    for pa in pas:
        dct = state[pa]
        m_lst.append(dct['exp_avg'].detach().flatten())
        v_lst.append(dct['exp_avg_sq'].detach().flatten())
        
    m:Tensor = cat(m_lst, dim=0)
    v:Tensor = cat(v_lst, dim=0)
    t:Tensor = dct['step']
    return m, v, t