from opt_funcs.base import OptUtils
from type_ import *

def read_state(optimizer:Optimizer,
               params:List[Tensor],
               startends,
               on_device:bool):
    from torch import no_grad,cat
    with no_grad:
        states = optimizer.state
        ms, vs, ts = [],[],[]
        dct = {'ms':ms, 'vs':vs, 'ts':ts}
        for start,end in startends:
            pas = params[start:end]
            m_lst = []
            v_lst = []
            # t_lst = []
            state = None
            for pa in pas:
                state = states[pa]
                m_lst.append(state['exp_avg'].flatten())
                v_lst.append(state['exp_avg_sq'].flatten())
                # t_lst.append(state['step'].item())
            m = cat(m_lst, 0)
            v = cat(v_lst, 0)
            t = state['step']
            if not on_device:
                m = m.to('cpu')
                v = v.to('cpu')
            ms.append(m)
            vs.append(v)
            ts.append(t)
    return dct

def read_coef(optimizer:Optimizer):
    state_dct = optimizer.state_dict()
    coefs = state_dct['param_groups']
    return coefs

def update_bp_states(flat_grads, params, startends, dLdws, state, backward_state:Dict[list], coef, device, train_lr):
    from torch import cat, zeros_like, no_grad, tensor, sqrt, zeros
    
    with no_grad():
        if 'dLdms' not in backward_state:
            dLdms = [zeros_like(dLdw) for dLdw in dLdws]
            backward_state['dLdms'] = dLdms
            dLdvs = [zeros_like(dLdw) for dLdw in dLdws]
            backward_state['dLdvs'] = dLdvs
        else:
            dLdms = backward_state['dLdms']
            dLdvs = backward_state['dLdvs']
        ms = state['ms']
        vs = state['vs']
        ts = state['ts']
        coefs = coef
        dLdgrads = []
        lr_grad = zeros(1)
        for (start, end), flat_grad, dLdw, dLdm, dLdv, m, v, t, coef in zip(startends, flat_grads, dLdws, dLdms, dLdvs, ms, vs, ts, coefs):
            gt = flat_grad.detach()
            lr = coef['lr']
            maximize = coef['maximize']
            beta1, beta2 = coef['betas']
            eps = coef['eps']
            weight_decay = coef['weight_decay']
            if m.device != device:
                m = m.to(device)
                v = v.to(device)
            omb1 = 1. - beta1
            omb1t = 1. - beta1**t
            omb2 = 1. - beta2
            omb2t = 1. - beta2**t
            rtvdomb2t = sqrt(v.div(omb2t))
            rtvdomb2tneps = rtvdomb2t.add(eps)
            if train_lr:
                a = m.div(omb1t).div(rtvdomb2t.add(eps))
                lr_grad.add_(dLdw.dot(a).mul(-1))
            if maximize:
                gt.mul_(-1.)
            if weight_decay!=0:
                w = cat([ele.flatten() for ele in params[start:end]])
                gt.add_(w.mul(weight_decay))
            dLdm.mul_(beta1).sub_(dLdw.div(rtvdomb2tneps), lr/omb1t)
            dLdv.mul_(beta2).add_(dLdw.mul(m).div(rtvdomb2t).div(rtvdomb2tneps.pow(2)), lr/2./omb1t/omb2t)
            gt.mul_(2*omb2).mul_(dLdv).add_(dLdm, omb1)
            if weight_decay != 0:
                dLdw.add_(gt, weight_decay)
            if maximize:
                gt.mul_(-1)
            dLdgrads.append(gt)
        return dLdgrads, lr_grad

            
        
            
        
        




    


        
         


