from type_ import *

class MEBPTT():

    def __init__(self):
        self.foward_function_handle:Callable=None
        self.forward_args_lst = []
        self.forward_kwargs_lst = dict()
        self.param_idx_in_model_lst = None
        self.backbone = None
        self.model_tape = None
        self.state_tape = None
        self.coef_tape = None
        self.params = None
        self.cur_idx = None
        self.dLdws = []
        self.opt_type:str=None
        self.device = None
        self.backward_state = dict()
        self.lr_grad:Tensor = torch.zeros(1).requires_grad_(False)

    def forward_loop(self,
                     num_steps:int,
                     num_tapes:int,
                     backbone:Module,
                     optimizer:Optimizer,
                     forward_function_handle:Callable,
                     forward_args_lst:List[list]=None,
                     forward_kwargs_lst:List[dict]=None,
                     tape_on_device:bool=True
                     ):
        '''
        the first positional arg of forward_function_handle must be 
        the inner model to be optimized.
        '''
        from opt_funcs import base
        from opt_funcs.base import inplace_backward
        from opt_funcs import adam
        from torch import no_grad, randn_like
        from torch.optim import Adam
        from functools import partial
        from . import utils
        #declarations:
        copy_mod_pa = base.copy_model_and_params
        forwardloss = utils.args_and_kwargs(forward_function_handle,
                                            forward_args_lst,
                                            forward_kwargs_lst)


        #preparations:
        #prepare model
        params, startends = base.read_params_from_opt(optimizer)
        self.device = params[0].device
        self.startends = startends
        if self.param_idx_in_model_lst is None:
            param_idx_in_model_lst = base.param_idx_in_model(params, backbone)
            self.param_idx_in_model_lst = param_idx_in_model_lst
        else:
            param_idx_in_model_lst = self.param_idx_in_model_lst
        #prepare optimizer
        if isinstance(optimizer, Adam):
            self.opt_type = 'adam'
            read_state = adam.read_state
            read_coef = adam.read_coef
        else:
            raise NotImplementedError
        zero_grad = optimizer.zero_grad
        step = optimizer.step
        #prepare args and kwargs

        model_tape:List[Tuple[Module, List[Tensor]]] = []
        state_tape:List[dict] = []
        coef_tape:List[dict] = []
        start_taping = num_steps - num_tapes
        #forward inner loop
        cur_idx = 0
        for idx in range(num_steps):
            zero_grad()
            tape = idx >= start_taping
            if tape:
                model_tape.append(copy_mod_pa(backbone, param_idx_in_model_lst, tape_on_device))
            else:
                model_tape.append(None)

            loss = forwardloss(backbone, idx)
            inplace_backward(loss, params)
            if idx==0:
                with no_grad():
                    for pa in params:
                        pa.grad.add_(randn_like(pa).mul(1e-10))
            step()
            if tape:
                state_tape.append(read_state(optimizer, params, startends, tape_on_device))
                coef_tape.append(read_coef(optimizer))
            else:
                state_tape.append(None)
                coef_tape.append(None)
            cur_idx += 1
        zero_grad()
        self.backbone = backbone
        self.model_tape = model_tape
        self.state_tape = state_tape
        self.coef_tape = coef_tape
        self.params = params
        self.cur_idx = cur_idx
        self.forward_args_lst = forward_args_lst
        self.forward_kwargs_lst = forward_kwargs_lst
        self.forward_function_handle = forward_function_handle
        return backbone, model_tape, state_tape, coef_tape, cur_idx


    def backward_metaloss(self,
                          meta_loss:Tensor,
                          weight:float,
                          params_:List[Tensor]=None,
                          startends_=None
                          ):
        from opt_funcs.base import inplace_backward, update_dLdw
        meta_loss = weight * meta_loss
        if params_ is None:
            params_ = self.params
        if startends_ is None:
            startends_ = self.startends
        inplace_backward(meta_loss, params_)
        update_dLdw(params_, self.dLdws, startends_)
        return None
    
    def backward_loop(self,
                      num_steps:int,
                      meta_params:List[Tensor],
                      forward_args_lst:List[list]=None,
                      forward_kwargs_lst:List[dict]=None,
                      cur_idx:int=None,
                      model_tape_=None,
                      state_tape_=None,
                      coef_tape_=None,
                      startends_=None,
                      tape_on_device:bool=True,
                      train_lr:bool=False
                      ):
        from . import utils
        from opt_funcs import adam, base
        from torch import cat
        if forward_args_lst is None:
            forward_args_lst = self.forward_args_lst
        if forward_kwargs_lst is None:
            forward_kwargs_lst = self.forward_kwargs_lst
        forwardloss = utils.args_and_kwargs(self.foward_function_handle,
                                            forward_args_lst,
                                            forward_kwargs_lst)
        get_grad = base.get_grads_with_graph
        backward_on = base.backward_on_params_and_meta_params
        update_dLdws = base.update_dLdw
        if tape_on_device:
            model_on_device = lambda model: model.to(self.device)
        else:
            model_on_device = lambda model: None
        if cur_idx is None:
            cur_idx = self.cur_idx
        if model_tape_ is None:
            model_tape_ = self.model_tape
        if state_tape_ is None:
            state_tape_ = self.state_tape
        if coef_tape_ is None:
            state_tape_ = self.coef_tape
        if startends_ is None:
            startends_ = self.startends
        if self.opt_type=='adam':
            update_bp_states = adam.update_bp_states
        else:
            raise NotImplementedError
        if train_lr:
            lr_grad = self.lr_grad
        
        dLdws = self.dLdws
        backward_state = self.backward_state
        device = self.device
        start_idx = cur_idx - num_steps
        if start_idx<0:
            raise AssertionError("more steps than forwards!")
        
        for idx in range(cur_idx-1, cur_idx-num_steps-1, -1):
            cur_idx -= 1
            model_n_params = model_tape_[cur_idx]
            if model_n_params is not None:
                model, params = model_n_params
                model_on_device(model)
            else:
                raise AssertionError('step {} was not taped!'.format(cur_idx))
            state = state_tape_[cur_idx]
            coef = coef_tape_[cur_idx]
            loss = forwardloss(model, idx)
            grads = get_grad(loss, params)
            flat_grads = [cat([grd.flatten() for grd in grads[start:end]]) for start, end in startends_]
            if train_lr:
                dLdgrads, lr_grad_ = update_bp_states(flat_grads, params, startends_, dLdws, state, backward_state, coef, device, True)
                lr_grad.add_(lr_grad_)
            else:
                dLdgrads, _ = update_bp_states(flat_grads, params, startends_, dLdws, state, backward_state, coef, device, False)
            backward_on(grads, dLdgrads, meta_params, params)
            update_dLdws(params, dLdws, startends_)
        return model, cur_idx



            

        
        

            
            



        


        
