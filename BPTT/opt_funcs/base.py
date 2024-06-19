from type_ import *


def read_params_from_opt(optimizer:Optimizer):
    '''
    unroll parameter groups in optimizer.param_groups into one list
    return a list of params and a list of group start-end positions
    '''
    param_groups = optimizer.param_groups
    attached_params = []
    group_startends:List[Tuple[int,int]] = []
    start_idx = 0
    for group in param_groups:
        pas = group['params']
        attached_params.extend(pas)
        end_idx = start_idx + len(pas)
        group_startends.append((start_idx, end_idx))
        start_idx = end_idx
    return attached_params, group_startends

def param_idx_in_model(params:List[Tensor], model:Module):
    '''
    returns a list of the size of params, indicating each param's idx in model.parameters()
    '''
    attached_params = params
    model_params = model.parameters()
    attpa2modpa_idxes = []
    paslen = len(model_params)
    for attpa in attached_params:
        for paidx,pa in enumerate(model_params):
            if attpa is pa:
                attpa2modpa_idxes.append(paidx)
                break
            if paidx+1 == paslen and not attpa is pa:
                raise AssertionError("model_params do not contain attached_params {}!".format(attpa))
    return attpa2modpa_idxes

def copy_model_and_params(model:Module, params_idx_lst:List[int]=None, on_device:bool=True):
    from copy import deepcopy
    copymodel = deepcopy(model)
    if not on_device:
        copymodel.to('cpu')
    model_params = list(copymodel.parameters())
    for pa in model_params:
        pa.grad = None
    if params_idx_lst is None:
        params = model_params
    else:
        params = [model_params[idx] for idx in params_idx_lst]
    return copymodel, params

def inplace_backward(loss:Tensor, params:List[Tensor]):
    from torch.autograd import backward
    backward(loss, inputs=params)
    return None

def get_grads_with_graph(loss:Tensor, params:List[Tensor]):
    from torch.autograd import grad
    grads = grad(loss, inputs=params, retain_graph=True, create_graph=True)
    return grads

def backward_on_meta_params(grads:List[Tensor], 
                            dLdgrads:List[Tensor], 
                            meta_params:List[Tensor]):
    from torch.autograd import backward
    from torch import cat
    grads_ = cat([ele.flatten() for ele in grads])
    dLdgrad_ = cat([ele.flatten() for ele in dLdgrads])
    backward(tensors=grads_,
             grad_tensors=dLdgrad_,
             inputs = meta_params)
    return None

def backward_on_params_and_meta_params(grads:List[Tensor],
                                       dLdgrads:List[Tensor],
                                       meta_params:List[Tensor],
                                       params:List[Tensor]):
    from torch.autograd import backward
    from torch import cat
    grads_ = cat([ele.flatten() for ele in grads])
    dLdgrad_ = cat([ele.flatten() for ele in dLdgrads])
    params.extend(meta_params)
    backward(tensors=grads_,
             grad_tensors=dLdgrad_,
             inputs=params)
    return None

def update_dLdw(params:List[Tensor], 
                dLdws:List[Tensor], 
                startends:List[Tuple[int,int]]):
    from torch import cat, no_grad
    with no_grad():
        if len(dLdws)==0:
            for start,end in startends:
                dLdws.append(cat([ele.grad.flatten() for ele in params[start:end]]))
        else:
            for (start, end), dLdw in zip(startends, dLdws):
                dLdw.add_(cat([ele.grad.flatten() for ele in params[start:end]]))
    return None






