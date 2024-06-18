def get_diff_opt(model, optimizer, tape_on_device, _attpa2modpa_idxes=None):
    from torch.optim import Adam
    from BPTT.diff_adam import DiffAdam
    if isinstance(optimizer, Adam):
        return DiffAdam(model, optimizer, tape_on_device, _attpa2modpa_idxes)
    else:
        raise NotImplementedError
