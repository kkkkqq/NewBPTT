
    
def get_optimizer(params, opt_name:str, **kwargs):
    from torch.optim import SGD, Adam
    if opt_name.lower()=='sgd':
        return SGD(params=params, **kwargs)
    elif opt_name.lower()=='adam':
        return Adam(params=params, **kwargs)
    else:
        raise NotImplementedError
    
