import torch
import torch.nn as nn
import copy
from type_ import *
import numpy as np

class DiffOptimizer():

    def __init__(self, 
                 model:Module, 
                 optimizer:Optimizer, 
                 state_tape_required:bool, 
                 tape_on_device:bool=True,
                 paidx_grpidx_map:List[List[Tuple[int,int]]]=None)->None:
        self.model:Module = model
        self.optimizer:Optimizer = optimizer
        if paidx_grpidx_map is None:
            self.paidx_grpidx_map:List[List[Tuple[int,int]]] = self.model_opt_mapping(model, optimizer)
        else:
            self.paidx_grpidx_map = paidx_grpidx_map
        self.device = next(model.parameters()).device
        self.tape_on_device:bool = tape_on_device
        self.model_tape:List[Module] = []
        self.states_tape:List[List[List[Dict[str,Tensor]]]] = []
        self.state_tape_required:bool=state_tape_required # 有一些优化器在反传时不需要states，比如SGD
        self._model_grads_after_backward:List[Tensor] = []
        self.dLdw_groups:List[Tensor] = []
        self.len_pa_groups:List[int] = []
        self._pa_groups_startend_lst:List[Tuple[int,int]] = []
        self.dLdw_groups, self.len_pa_groups, self._pa_groups_startend_lst = self.prepare_dLdw()
        self.dLdgrad_groups = []
        self.cur_idx:int = 0
        self._forward_function:Callable = None
        self._forward_kwargs:dict = None
        
    ### major wrapper functions
    def forward_loop(self, 
                     forward_function:Callable, 
                     num_steps:int,
                     num_taped:int=None,
                     forward_kwargs:dict=dict()):
        """
        wrapper function for the inner forward loop.\\
        `forward_function`:a function that takes in `step_idx` and `backbone` and returns `loss`.\\
        `num_steps`: number for inner forward steps to perform.\\
        `num_taped`: number of taped steps at the end. \\
        `forward_kwargs`: kwargs that will be passed into forward_function apart from `step_idx` and `backbone`.
        """
        self._forward_function = forward_function
        self._forward_kwargs = forward_kwargs
        zero_grad = self.zero_grad
        backward = self.backward
        step = self.step
        for idx_ in range(num_steps):
            zero_grad()
            step_idx = self.cur_idx
            loss = forward_function(step_idx=step_idx, backbone=self.model, **forward_kwargs)
            backward(loss, False, False, False, True)
            taped = not num_taped is not None and idx_ < num_steps-num_taped
            step(taped)
        return None
    
    def backward_loop(self, num_steps:int, meta_params:List[Tensor]):
        '''
        Wrapper function for backward inner loop.\\
        It computes the meta grads on meta_params for each step and accumulates
        each of them on .grad of each meta param.
        '''
        roll_back = self.roll_back
        zero_grad = self.zero_grad
        backward = self.backward
        forward_function = self._forward_function
        forward_kwargs = self._forward_kwargs
        backprop_step = self.backprop_step
        for _ in range(num_steps):
            model = roll_back()
            zero_grad()
            loss = forward_function(step_idx=self.cur_idx, backbone=model, **forward_kwargs)
            backward(loss, True, True, False, False)
            backprop_step(meta_params, True, True)
        return None


            





    ### helper function for init
    @staticmethod
    def model_opt_mapping(model:Module, optimizer:Optimizer)->List[List[Tuple[int,int]]]:
        '''helper function for establishing a map from idxes of params in model to idxes of groups and idxes in groups.'''
        params = model.parameters()
        param_groups = optimizer.param_groups
        paidx_grpidx_map = []
        for pa in params:
            #对model的每个param
            grpidx_lst:List[Tuple[int,int]] = []
            for gidx, group in enumerate(param_groups):
                #对opt的每个param groups
                #检查pa是否在这个groups里并返回它的idx
                g_pas:List[Tensor] = group['params']
                pos_lst = [tsr is pa for tsr in g_pas]
                try:
                    pos_idx = pos_lst.index(True)
                except:
                    continue
                #如果在，就记录pa所在的groupidx和posidx
                grpidx_lst.append((gidx, pos_idx))
            #在paidx_grpidx_map的这个param对应的idx处记录[(grp_idx, pos_idx), (grp_idx, pos_idx),...]
            paidx_grpidx_map.append(grpidx_lst)
        return paidx_grpidx_map

    def prepare_dLdw(self):
        from numpy import sum
        from torch import zeros
        self.dLdw_groups = []
        self.len_pa_groups = []
        self._pa_groups_startend_lst = []
        start_idx = 0
        for group in self.optimizer.param_groups:
            params:List[Tensor] = group['params']
            full_len = sum([pa.numel() for pa in params])
            self.dLdw_groups.append(zeros(full_len, device=self.device))
            grouplen = len(params)
            self.len_pa_groups.append(grouplen)
            end_idx = start_idx + grouplen
            self._pa_groups_startend_lst.append((start_idx, end_idx))
            start_idx = end_idx
        return self.dLdw_groups, self.len_pa_groups, self._pa_groups_startend_lst
        
    ### functions resembling optimizers and their helper functions
    def step(self, taped:bool, closure:Callable=None):
        self._pre_step(self, taped)
        self.optimizer.step(closure=closure)
        self._post_step(self, taped)
        return None

    def _pre_step(self, taped):
        from copy import deepcopy
        if taped:
            copymodel = deepcopy(self.model)
            if not self.tape_on_device:
                copymodel.to('cpu')
            self.model_tape.append(copymodel)
        else:
            self.model_tape.append(None)
        return None
    
    def _post_step(self, taped):
        if self.state_tape_required and taped:
            grouped_states_lst = self.copy_state(self.optimizer, not self.tape_on_device)
            self.states_tape.append(grouped_states_lst)
        else:
            self.states_tape.append(None)
        self.cur_idx += 1
        return None
    
    def zero_grad(self, set_to_none:bool=True):
        self.optimizer.zero_grad(set_to_none)
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
                    sts = deepcopy(state[pa])
                    for k,v in sts.items():
                        if isinstance(v, Tensor):
                            sts[k] = v.to('cpu')
                    pas_states_lst.append(sts)
                grouped_states_lst.append(pas_states_lst)
        else:
            for pa_group in param_groups:
                pas = pa_group['params']
                pas_states_lst = []
                for pa in pas:
                    sts = deepcopy(state[pa])
                    pas_states_lst.append(sts)
                grouped_states_lst.append(pas_states_lst)
        return grouped_states_lst
    
    ### methods specific to DiffOptimizer
    def backward(self, 
                 loss:Tensor, 
                 retain_graph:bool=False, 
                 create_graph:bool=False, 
                 accum_grad:bool=False,
                 update_grad:bool=True):
        '''
        call this function before calling self.step().
        backward function, takes in loss:Tensor and updates the grads in param_groups
        '''
        from torch.autograd import grad
        params:List[Tensor] = []
        for group in self.optimizer.param_groups:
            pas = group['params']
            params.extend(pas)
        grads = grad(outputs=loss,
                     inputs=params,
                     retain_graph=retain_graph,
                     create_graph=create_graph)
        self._model_grads_after_backward = grads
        if update_grad:
            if accum_grad:
                with torch.no_grad():
                    for gr,pa in zip(grads, params):
                        if pa.grad is None:
                            pa.grad = gr
                        else:
                            pa.grad += gr
            else:
                with torch.no_grad():
                    for gr,pa in zip(grads, params):
                        pa.grad = gr
        return None
    
    def meta_backward(self, meta_loss:Tensor, weight:float=1., refresh_bp:bool=False):
        '''
        Call this function after the computation of meta loss.\\
        If meta loss is computed in batches, call this function
        multiple times specifying weight. The resulting gradient
        will be a weighted sum.
        '''
        from torch import cat, no_grad
        meta_loss = meta_loss.mul(weight)
        self.backward(loss=meta_loss, 
                      retain_graph=False, 
                      create_graph=False,
                      accum_grad=False,
                      update_grad=False)
        grads:List[Tensor] = self._model_grads_after_backward
        grad_groups:List[List[Tensor]] = [grads[startend[0]:startend[1]] for startend in self._pa_groups_startend_lst]
        with no_grad():
            if refresh_bp:
                self.dLdw_groups = [cat([grad_.flatten() for grad_ in grads_]) for grads_ in grad_groups]
            else:
                for grads_, dLdw in zip(grad_groups, self.dLdw_groups):
                    dLdw.add_(cat([grad_.flatten() for grad_ in grads_]))
    
    def roll_back(self):
        """
        roll back to last step.
        Returns the backbone model at last step.
        """
        step_idx = self.cur_idx - 1
        model = self.at_step(step_idx)
        return model
    
    def at_step(self, step_idx:int, change_state:bool=False)->Module:
        '''
        Returns taped model, and attach the parameter groups of self.optimizer to it.\\
        Note that this function does not change self.optimizer.states unless change_state
        is True.
        '''
        if change_state:
            raise NotImplementedError("haven't implemented changing optimizer.state to step_idx")
        
        model = self.model_tape[step_idx]
        if not self.tape_on_device:
            model.to(self.device)
            self.model.to('cpu')
        self.cur_idx = step_idx
        self.model = model
        if model is None:
            raise AssertionError("step {} was not taped!".format(step_idx))
        paidx_grpidx_map = self.paidx_grpidx_map
        param_groups = self.optimizer.param_groups
        for grpidxlst, param in zip(paidx_grpidx_map, model.parameters()):
            if len(grpidxlst)!=0:
                for grpidx in grpidxlst:
                    param_groups[grpidx[0]]['params'][grpidx[1]] = param
        return model

    def backprop_step(self,
                      meta_params:List[Tensor],
                      accum_grad:bool=True,
                      update_bp_states:bool=True):
        """
        Take one step backward and compute the meta gradients for meta_params.\\
        If accum_grad, the meta gradients will be added to meta_params[:].grad;
        if not accum_grad, the meta gradients will replace meta_params[:].grad.\\
        The backbone model must have been roll_backed and forwarded and backwarded
        with its param.grad ready before calling backprop_step.
        """
        from torch import no_grad
        if update_bp_states:
            #print('memory before update state', torch.cuda.memory_allocated(0))
            self.update_backprop_state()
            #print('memory after update state', torch.cuda.memory_allocated(0))
            meta_grads = self.backprop_meta_params(meta_params, True)
            #print('memory after backprop metaparams', torch.cuda.memory_allocated(0))
        else:
            meta_grads = self.backprop_meta_params(meta_params)
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

    def update_backprop_state(self):
        """
        Optimizer-specific backprop update function, in-place modify all backprop states.\\
        The dLdw_groups is only partially computed, it requires a further dLdgrad*dgraddw
        to be later computed and added to itself.\\
        Must be precedented by backward.
        """
        raise NotImplementedError
    
    def backprop_meta_params(self, meta_params:List[Tensor], update_dLdw:bool=True):
        """
        Compute meta gradients for params in meta_params.\\
        Must be precedented by update_backprop_state.\\
        """
        from torch import no_grad
        from torch.autograd import grad
        params = []
        opt = self.optimizer
        flatten = self.flatten
        dLdw_groups = self.dLdw_groups
        if update_dLdw:
            pa_groups_startend_lst = self._pa_groups_startend_lst
            #把meta_params加到params后面
            for group in opt.param_groups:
                pas = group['params']
                params.extend(pas)
        params.extend(meta_params)
        grads = self._model_grads_after_backward
        grads = flatten(grads)
        dLdgrad = flatten(self.dLdgrad_groups)
        meta_grads = grad(outputs=grads,
                          inputs=params,
                          grad_outputs=dLdgrad,)
        self._tracked_grads = None
        with no_grad():
            if update_dLdw:
                for dLdw, startend in zip(dLdw_groups, pa_groups_startend_lst):
                    dLdw.add_(flatten(meta_grads[startend[0]:startend[1]]))
                meta_grads = meta_grads[pa_groups_startend_lst[-1][1]:]
        return meta_grads

    @staticmethod
    def flatten(tsr_lst:List[Tensor]):
        from torch import cat
        return cat([tsr.flatten() for tsr in tsr_lst])

        

        





    
