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
        self._pa_groups_startend_lst = []
        self.dLdw_groups, self.len_pa_groups, self._pa_groups_startend_lst = self.prepare_dLdw()
        self.cur_idx:int = 0
        
        

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
        meta_loss.mul_(weight)
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


            

        

        





    
