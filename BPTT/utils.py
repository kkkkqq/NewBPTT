def get_diff_opt(model, optimizer, tape_on_device, paidx_grpidx_map):
    from torch.optim import Adam
    from BPTT.diff_adam import DiffAdam
    if isinstance(optimizer, Adam):
        return DiffAdam(model, optimizer, tape_on_device, paidx_grpidx_map)
    else:
        raise NotImplementedError
