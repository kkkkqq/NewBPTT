from BPTT.diff_adam import DiffOptimizer
from type_ import *
from modules.utils import get_module
from modules.basemodule import BaseModule
from dataset.baseset import ImageDataSet

class InnerLoop():
    def __init__(self,
                 batch_function:Callable,
                 inner_module_args:dict,
                 inner_model_args:dict,
                 inner_opt_args:dict,
                 inner_batch_size:int,
                 device='cuda'):
        self.batch_function = batch_function # takes (batch_idx, batch_size, *args, **kwargs)
        self.inner_module_args = inner_module_args
        self.device = torch.device(device)
        self.inner_module:BaseModule = get_module(**self.inner_module_args)
        self.inner_model_args = inner_model_args
        self.inner_opt_args = inner_opt_args
        self.inner_batch_size = inner_batch_size
        self._attpa2modpa_idxes = None
        self.backbone = None
        self.diff_opt = None

    def forward_function(self, backbone:Module, step_idx:int,  **batch_kwargs):
        batch_out = self.batch_function(batch_idx=step_idx, batch_size=self.inner_batch_size, **batch_kwargs)
        forward_loss_out = self.inner_module.forward_loss(backbone, *batch_out)
        return forward_loss_out[0]
    
    def meta_loss_and_backward(self, backbone:Module, **meta_loss_kwargs)->float:
        raise NotImplementedError
    
    def loop(self,
             num_forward:int,
             num_backward:int,
             meta_params:List[Tensor],
             batch_kwargs:dict=dict(),
             meta_loss_kwargs:dict=dict())->float:
        from models.utils import get_model
        from utils import get_optimizer
        from BPTT.diff_optim import DiffOptimizer
        from BPTT.utils import get_diff_opt
        backbone = get_model(**self.inner_model_args)
        backbone.to(self.device)
        opt = get_optimizer(backbone.parameters(), **self.inner_opt_args)
        diffopt = get_diff_opt(backbone, opt, True, self._attpa2modpa_idxes)
        self._attpa2modpa_idxes = diffopt.attpa2modpa_idxes
        self.backbone = backbone
        self.diff_opt = diffopt
        forward_args_lst = None
        forward_kwargs_lst = [{'step_idx':idx} for idx in range(num_forward)]
        for dct in forward_kwargs_lst:
            dct.update(batch_kwargs)

        diffopt.forward_loop(self.forward_function,
                             num_forward,
                             num_backward,
                             forward_args_lst,
                             forward_kwargs_lst)
        meta_loss_item = self.meta_loss_and_backward(backbone=backbone, **meta_loss_kwargs)
        diffopt.backward_loop(num_backward, meta_params)
        return meta_loss_item

        
             
        