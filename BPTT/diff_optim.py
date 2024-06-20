import torch
import torch.nn as nn
import copy
from type_ import *
import numpy as np

class DiffOptimizer():

    def __init__(self, 
                 model:Module, 
                 optimizer:Optimizer, 
                 tape_on_device:bool=True,
                 _attpa2modpa_idxes:List[int]=None
                 )->None:
        self.model:Module = model
        self.optimizer:Optimizer = optimizer
        self.opt_state = optimizer.state
        self.opt_param_groups = optimizer.param_groups
        
        param_groups = optimizer.param_groups
        attached_params:List[Tensor] = []
        group_startends:List[Tuple[int,int]] = []
        attached_params, group_startends = self.unroll_param_groups(param_groups)
        attpa2modpa_idxes:List[int] = []
        model_params = list(model.parameters())
        if _attpa2modpa_idxes is None:
            attpa2modpa_idxes = self.attpa2modpa(attached_params, model_params)
        else:
            attpa2modpa_idxes = _attpa2modpa_idxes
        self.attached_params = attached_params
        self.group_startends = group_startends
        self.attpa2modpa_idxes = attpa2modpa_idxes
        self.device = model_params[0].device
        self.tape_on_device:bool = tape_on_device
        self.model_tape:List[Module] = []
        self.states_tape_dct:Dict[str, List[List[Tensor]]] = dict()#for adam, {'m':flat_groups_tape, 'v':flat_groups_tape}
        self.coefs_tape_dct:Dict[str, List[list]] = {'lr_groups':None}# tapes for 'lr_groups', 'eps_groups', etc. coefs stored in groups.
        self.state_vars:Dict[str,List[Tensor]] = {'dLdw_groups':[], 'dLdgrad_groups':[]}
        self.group_coefs:Dict[str,list] = {'lr_groups':[]}
        self.cur_idx:int = 0
        self.forward_function:Callable = None
        self.forward_args_lst = None
        self.forward_kwargs_lst = None
        self.lr_grad:Tensor = torch.zeros(1, device=self.device)

    @staticmethod
    def unroll_param_groups(param_groups:dict):
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
    
    @staticmethod
    def attpa2modpa(attached_params, model_params):
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
    
    @staticmethod
    def group_params(params:List[Tensor], group_startends:List[Tuple[int,int]]):
        return [params[start:end] for start, end in group_startends]
        
    ### major wrapper functions
    def forward_loop(self, 
                     forward_function:Callable, 
                     num_steps:int,
                     num_taped:int=None,
                     forward_args_lst:List[list]=None,
                     forward_kwargs_lst:List[dict]=None,
                     ):
        """
        wrapper function for the inner forward loop.\\
        `forward_function`: its first positional arg must be the model.\\
        `num_steps`: number for inner forward steps to perform.\\
        `num_taped`: number of taped steps at the end. \\
        `forward_kwargs`: kwargs that will be passed into forward_function apart from `step_idx` and `backbone`.
        """
        
        self.forward_function = forward_function
        self.forward_args_lst = forward_args_lst
        self.forward_kwargs_lst = forward_kwargs_lst
        forwardloss = args_and_kwargs(forward_function, forward_args_lst, forward_kwargs_lst)
        tape_on_device = self.tape_on_device
        zero_grad = self.zero_grad
        backward = self.backward
        step = self.step
        params = self.attached_params
        state_tapes_dct = self.states_tape_dct
        coef_tapes_dct = self.coefs_tape_dct
        model_tape = self.model_tape
        model = self.model
        for idx_ in range(num_steps):
            zero_grad()
            step_idx = self.cur_idx
            loss = forwardloss(model, idx_)
            backward(loss, params, True, True, False, False)
            taped = idx_ >= num_steps-num_taped or num_taped is None
            step(taped, None, tape_on_device, model_tape, state_tapes_dct, coef_tapes_dct)
        print('{} inner forward steps'.format(num_steps))
        return None
    
    def meta_backward(self, 
                      meta_loss:Tensor, 
                      weight:float=1., 
                      dLdw_groups:List[Tensor]=None, 
                      params:List[Tensor]=None, 
                      group_startends:List[Tuple[int,int]]=None,
                      _delete_optimizer:bool=True,
                      _backward_handle:Callable=None):
        '''
        Call this function after the computation of meta loss.\\
        If meta loss is computed in batches, call this function
        multiple times specifying weight. The resulting gradient
        will be a weighted sum.//
        If dLdw_groups is None, backward in-place update will be on 
        self.state_var['dLdw_groups']; 
        otherwise it in-place modifies dLdw_groups and returns it. 
        '''
        from torch import cat, no_grad
        from BPTT.jits import flatten
        meta_loss = meta_loss.mul(weight)
        if _backward_handle is None:
            self_backward = self.backward
        else:
            self_backward = _backward_handle
        grads = self_backward(loss=meta_loss, 
                              params = params,
                              retain_graph=False, 
                              create_graph=False,
                              accum_grad=False,
                              update_grad=False)
        if group_startends is None:
            group_startends = self.group_startends
        grad_groups:List[List[Tensor]] = [grads[start:end] for start,end in group_startends]
        with no_grad():
            if dLdw_groups is None:
                dLdw_groups = self.state_vars['dLdw_groups']
            if len(dLdw_groups)!=0:
                for grads_, dLdw in zip(grad_groups, dLdw_groups):
                    dLdw.add_(flatten(grads_))
            else:
                dLdw_groups.clear()
                for grads_ in grad_groups:
                    dLdw_groups.append(flatten(grads_))
        if _delete_optimizer:
            self.optimizer = None
            self.opt_state = None
            self.opt_param_groups = None
            # self.attached_params = None
            
        
        return dLdw_groups

    def backward_loop(self, 
                      num_steps:int, 
                      meta_params:List[Tensor],
                      trainable_lr:bool=False,
                      forward_function:Callable=None,
                      forward_args_lst:List[list]=None,
                      forward_kwargs_lst:List[dict]=None):
        '''
        Wrapper function for backward inner loop.\\
        It computes the meta grads on meta_params for each step and accumulates
        each of them on .grad of each meta param.
        '''
        roll_back = self.roll_back
        zero_grad = self.zero_grad
        backward = self.backward
        if forward_function is None:
            forward_function = self.forward_function
        if forward_args_lst is None:
            forward_args_lst = self.forward_args_lst
        if forward_kwargs_lst is None:
            forward_kwargs_lst = self.forward_kwargs_lst
        forwardloss = args_and_kwargs(forward_function, forward_args_lst, forward_kwargs_lst)
        backprop_step = self.backprop_step
        state_vars = self.state_vars
        coefs_tape_dct = self.coefs_tape_dct
        attpa2modpa_idxes = self.attpa2modpa_idxes
        coef_tapes = coefs_tape_dct.items()
        state_tapes = self.states_tape_dct.items()
        group_startends = self.group_startends
        for _ in range(num_steps):
            cur_idx, model, attached_params = roll_back(attpa2modpa_idxes=attpa2modpa_idxes)
            group_coefs = {coef:coef_tape[cur_idx] for coef, coef_tape in coef_tapes}
            for state_key, state_tape in state_tapes:
                state_vars[state_key] = state_tape[cur_idx]
            zero_grad(True, attached_params)
            loss = forwardloss(model, cur_idx)
            grads = backward(loss=loss, 
                            params=attached_params, 
                            create_graph=True, 
                            retain_graph = True, 
                            accum_grad=False, 
                            update_grad=False)
            backprop_step(grads=grads,
                          attached_params=attached_params,
                          group_startends=group_startends,
                          meta_params=meta_params, 
                          update_bp_states=True,
                          state_vars=state_vars,
                          group_coefs=group_coefs,
                          accum_grad=True,
                          train_lr=trainable_lr)

        return None


            





    ### helper function for init
    # @staticmethod
    # def model_opt_mapping(model:Module, optimizer:Optimizer)->List[List[Tuple[int,int]]]:
    #     '''helper function for establishing a map from idxes of params in model to idxes of groups and idxes in groups.'''
    #     params = model.parameters()
    #     param_groups = optimizer.param_groups
    #     paidx_grpidx_map = []
    #     for pa in params:
    #         #对model的每个param
    #         grpidx_lst:List[Tuple[int,int]] = []
    #         for gidx, group in enumerate(param_groups):
    #             #对opt的每个param groups
    #             #检查pa是否在这个groups里并返回它的idx
    #             g_pas:List[Tensor] = group['params']
    #             pos_lst = [tsr is pa for tsr in g_pas]
    #             try:
    #                 pos_idx = pos_lst.index(True)
    #             except:
    #                 continue
    #             #如果在，就记录pa所在的groupidx和posidx
    #             grpidx_lst.append((gidx, pos_idx))
    #         #在paidx_grpidx_map的这个param对应的idx处记录[(grp_idx, pos_idx), (grp_idx, pos_idx),...]
    #         paidx_grpidx_map.append(grpidx_lst)
    #     return paidx_grpidx_map

    # def prepare_dLdw(self):
    #     from numpy import sum
    #     from torch import zeros
    #     self.dLdw_groups = []
    #     self.len_pa_groups = []
    #     self._pa_groups_startend_lst = []
    #     start_idx = 0
    #     for group in self.optimizer.param_groups:
    #         params:List[Tensor] = group['params']
    #         full_len = sum([pa.numel() for pa in params])
    #         self.dLdw_groups.append(zeros(full_len, device=self.device))
    #         grouplen = len(params)
    #         self.len_pa_groups.append(grouplen)
    #         end_idx = start_idx + grouplen
    #         self._pa_groups_startend_lst.append((start_idx, end_idx))
    #         start_idx = end_idx
    #     return self.dLdw_groups, self.len_pa_groups, self._pa_groups_startend_lst
        
    ### functions resembling optimizers and their helper functions
    def step(self, 
             taped:bool, 
             closure:Callable=None, 
             tape_on_device:bool=True,
             model_tape:List[Module]=None,
             state_tapes_dct:Dict[str,List[Tensor]]=None,
             coef_tapes_dct:Dict[str,List[list]]=None
            ):
        if model_tape is None:
            model_tape = self.model_tape
        self._pre_step(taped, tape_on_device, model_tape)
        self.optimizer.step(closure=closure)
        if state_tapes_dct is None:
            state_tapes_dct = self.states_tape_dct
        if coef_tapes_dct is None:
            coef_tapes_dct = self.coefs_tape_dct
        self._post_step(taped, tape_on_device, state_tapes_dct, coef_tapes_dct)
        self.cur_idx+=1
        return None

    def _pre_step(self, taped:bool, tape_on_device:bool, model_tape:List[Module]=None):
        '''
        by default tapes model copy, override for other behavior
        '''
        from copy import deepcopy
        if taped:
            copymodel = deepcopy(self.model)
            for pa in copymodel.parameters():
                pa.grad = None
            if not tape_on_device:
                copymodel.to('cpu')
            model_tape.append(copymodel)
        else:
            model_tape.append(None)
        return None
    
    def _post_step(self, 
                   taped:bool, 
                   tape_on_device:bool,
                   state_tapes_dct:Dict[str,List[Tensor]]=None,
                   coef_tapes_dct:Dict[str,list]=None):
        '''
        must override for model_specific behavior
        '''
        raise NotImplementedError
        # if self.state_tape_required and taped:
        #     grouped_states_lst = self.copy_state(self.optimizer, not self.tape_on_device)
        #     self.states_tape.append(grouped_states_lst)
        # else:
        #     self.states_tape.append(None)
        # self.cur_idx += 1
        return None
    
    def zero_grad(self, set_to_none:bool=True, params:List[Tensor] = None):
        from torch import zeros_like, no_grad
        if params is None:
            self.optimizer.zero_grad(set_to_none)
        else:
            if set_to_none:
                for param in params:
                    param.grad = None
            else:
                with no_grad():
                    for param in params:
                        param.grad = zeros_like(param)
        return None
    
    @staticmethod
    def copy_state(opt:Optimizer, to_cpu:bool=False)->List[List[Dict[str, Tensor]]]:
        """
        returns a list of list of copied current states of each params in each param_groups in opt.
        """
        from copy import deepcopy
        state = opt.state
        param_groups = opt.param_groups
        grouped_states_lst = []
        if to_cpu:
            for pa_group in param_groups:
                pas = pa_group['params']
                pas_states_lst = []
                for pa in pas:
                    pas_states_lst.append({key:val.detach().clone().cpu() if isinstance(val, Tensor) else deepcopy(val) for key, val in state[pa].items()})
                grouped_states_lst.append(pas_states_lst)
        else:
            for pa_group in param_groups:
                pas = pa_group['params']
                pas_states_lst = []
                for pa in pas:
                    pas_states_lst.append({key:val.detach().clone() if isinstance(val, Tensor) else deepcopy(val) for key, val in state[pa].items()})
                grouped_states_lst.append(pas_states_lst)
        return grouped_states_lst
    
    ### methods specific to DiffOptimizer
    def backward(self, 
                 loss:Tensor, 
                 params:List[Tensor]=None,
                 update_grad:bool=True,
                 accum_grad:bool=False,
                 retain_graph:bool=False, 
                 create_graph:bool=False, 
                 )->List[Tensor]:
        '''
        call this function before calling self.step().
        backward function, takes in loss:Tensor and updates the grads in param_groups
        '''
        from torch.autograd import grad, backward as torchbackward
        if params is None:
            params = self.attached_params
        if update_grad:
            if not accum_grad:
                for pa in params:
                    pa.grad = None
            torchbackward(loss, inputs=params, retain_graph=retain_graph, create_graph=create_graph)
            return None
        else:
            grads = grad(outputs=loss,
                        inputs=params,
                        retain_graph=retain_graph,
                        create_graph=create_graph)
            return grads
    
    
    
    def roll_back(self, attpa2modpa_idxes:List[int]=None):
        """
        cur_idx -= 1,
        returns cur_idx, model at last step, and its parameters attached to optimizer
        """
        self.cur_idx -= 1
        step_idx = self.cur_idx
        if self.tape_on_device:
            model = self.taped_model_at_step(step_idx)
            self.model = model
        else:
            model = self.taped_model_at_step(step_idx, True)
            self.model.to('cpu')
            self.model = model
        params = list(model.parameters())
        if attpa2modpa_idxes is None:
            attpa2modpa_idxes = self.attpa2modpa_idxes
        attached_params = [params[idx] for idx in attpa2modpa_idxes]
        self.attached_params = attached_params
        return step_idx, model, attached_params
    
    def taped_model_at_step(self, step_idx:int, resume_device:bool=False)->Module:
        '''
        Returns a taped model
        '''
        model = self.model_tape[step_idx]
        if model is None:
            raise AssertionError("step {} was not taped!".format(step_idx))
        if resume_device:
            model.to(self.device)
        return model

    def backprop_step(self,
                      grads:List[Tensor],
                      attached_params:List[Tensor],
                      group_startends:List[Tuple[int,int]],
                      meta_params:List[Tensor],
                      update_bp_states:bool=True,
                      state_vars:Dict[str,List[Tensor]]=None,
                      group_coefs:Dict[str, List[Any]]=None,
                      accum_grad:bool=True,
                      train_lr:bool=False):
        """
        Compute the meta gradients for meta_params.\\
        If accum_grad, the meta gradients will be added to meta_params[:].grad;
        if not accum_grad, the meta gradients will replace meta_params[:].grad.\\
        The backbone model must have been roll_backed and forwarded and backwarded
        before calling backprop_step.\\
        If update_bp_states, state_vars and group_coefs must not be None.
        """
        from torch import no_grad
        from BPTT.jits import flatten
        if state_vars is None:
            state_vars = self.state_vars
        if group_coefs is None:
            group_coefs = self.group_coefs
        if update_bp_states:
            #print('memory before update state', torch.cuda.memory_allocated(0))
            flat_grads = [flatten(grads[start:end]) for start,end in group_startends]
            self.update_backprop_state(train_lr=train_lr, 
                                       params=attached_params, 
                                       grads=flat_grads,
                                       group_startends=group_startends,
                                       tape_on_device=self.tape_on_device,
                                       **state_vars, 
                                       **group_coefs)
            #print('memory after update state', torch.cuda.memory_allocated(0))
        meta_grads = self.backprop_meta_params(grads = flat_grads,
                                               meta_params = meta_params, 
                                               dLdgrad_groups = state_vars['dLdgrad_groups'],
                                               update_dLdw = True,
                                               attached_params=attached_params,
                                               group_startends=group_startends,
                                               dLdw_groups = state_vars['dLdw_groups'])
            
        with no_grad():
            if accum_grad:
                for megr,mepa in zip(meta_grads, meta_params):
                    if mepa.grad is None:
                        mepa.grad = megr
                    else:
                        mepa.grad.add_(megr)
            else:
                for megr,mepa in zip(meta_grads, meta_params):
                    mepa.grad = megr
        return meta_grads

    def update_backprop_state(self, 
                              params:List[Tensor],
                              grads:List[Tensor],
                              group_startends:List[Tuple[int,int]],
                              train_lr:bool,
                              tape_on_device:bool,
                              **kwargs)->None:
        """
        Optimizer-specific backprop update function, in-place modify all backprop state_vars.\\
        The kwargs are keys and vals in state_vars and group_coefs. It must include dLdw_groups,
        dLdgrad_groups, lr_groups. The rest of them depend on the specific optimizer.\\
        The dLdw_groups is only partially computed, it requires a further dLdgrad*dgraddw
        to be later computed and added to itself.\\
        Must be precedented by backward.\\
        """
        raise NotImplementedError
    
    def backprop_meta_params(self,
                             grads:List[Tensor],
                             meta_params:List[Tensor], 
                             dLdgrad_groups:List[Tensor],
                             update_dLdw:bool=True, 
                             attached_params:List[Tensor]=None,
                             group_startends:List[Tuple[int,int]]=None,
                             dLdw_groups:List[Tensor]=None,
                             ):
        """
        Compute meta gradients for params in meta_params.\\
        Must be precedented by update_backprop_state.\\
        If update_dLdw, will in-place update dLdw_groups.
        """
        from torch import no_grad, cat
        from torch.autograd import grad
        from BPTT.jits import flatten
        params = []
        if update_dLdw:
            params.extend(attached_params)
        params.extend(meta_params)
        grads = flatten(grads)
        dLdgrad = flatten(dLdgrad_groups)
        meta_grads = grad(outputs=grads,
                          inputs=params,
                          grad_outputs=dLdgrad,)
        with no_grad():
            if update_dLdw:
                for dLdw, (start,end) in zip(dLdw_groups, group_startends):
                    dLdw.add_(flatten(meta_grads[start:end]))
                meta_grads = meta_grads[group_startends[-1][-1]:]
        return meta_grads


        

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





    
