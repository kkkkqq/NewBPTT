def get_diff_opt(model, optimizer, tape_on_device, _attpa2modpa_idxes=None):
    from torch.optim import Adam
    from BPTT.diff_adam import DiffAdam
    if isinstance(optimizer, Adam):
        return DiffAdam(model, optimizer, tape_on_device, _attpa2modpa_idxes)
    else:
        raise NotImplementedError

def args_and_kwargs(function_handle, args_lst, kwargs_lst):
    none_args = args_lst is None
    none_kwargs = kwargs_lst is None
    
    if none_args and none_kwargs:
        forwardloss = lambda backbone, idx: function_handle(backbone)
    elif none_args and not none_kwargs:
        forwardloss = lambda backbone, idx: function_handle(backbone, **kwargs_lst[idx])
    elif not none_args and none_kwargs:
        forwardloss = lambda backbone, idx: function_handle(backbone, *args_lst[idx])
    else:
        forwardloss = lambda backbone, idx: function_handle(backbone, *args_lst[idx], **kwargs_lst[idx])
    return forwardloss