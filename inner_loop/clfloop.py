from BPTT.diff_adam import DiffOptimizer
from type_ import *
from modules.utils import get_module
from modules.basemodule import BaseModule
from dataset.baseset import ImageDataSet
from torch.utils.data import DataLoader
from inner_loop.baseloop import InnerLoop

class CLFInnerLoop(InnerLoop):
    def __init__(self,
                 batch_function:Callable,
                 inner_module_args:dict,
                 inner_model_args:dict,
                 inner_opt_args:dict,
                 inner_batch_size:int,
                 external_module_args:dict,
                 external_batch_size:int,
                 data_per_loop:int,
                 real_dataset:ImageDataSet,
                 device='cuda'):
        super().__init__(batch_function=batch_function,
                         inner_module_args=inner_module_args,
                         inner_model_args=inner_model_args,
                         inner_opt_args=inner_opt_args,
                         inner_batch_size=inner_batch_size,
                         device=device)
        self.external_module_args = external_module_args
        self.external_module:BaseModule = get_module(**external_module_args)
        self.data_per_loop = data_per_loop
        self.external_batch_size = external_batch_size
        self.real_loader = DataLoader(real_dataset.dst_train, external_batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
        self.loader_iter = iter(self.real_loader)

    # def forward_function(self, step_idx:int, backbone:Module, batch_kwargs:dict):
    #     batch_out = self.batch_function(batch_idx=step_idx, batch_size=self.inner_batch_size, **batch_kwargs)
    #     loss = self.inner_module.forward_loss(backbone=backbone, *batch_out)
    #     return loss
    
    def meta_loss_and_backward(self, backbone:Module, **meta_loss_kwargs)->float:
        num_data = 0
        meta_loss_item = 0.
        device = self.device
        data_per_loop = self.data_per_loop
        diff_opt = self.diff_opt
        forwardloss = self.external_module.forward_loss
        metabackward = diff_opt.meta_backward
        dLdw_groups_ = diff_opt.state_vars['dLdw_groups']
        params_ = diff_opt.attached_params
        group_startends_ = diff_opt.group_startends
        backward_handle_ = diff_opt.backward
        real_loader = self.real_loader
        if not real_loader.drop_last:
            raise AssertionError("the dataloader must have drop_last=True, otherwise weight for each batch will be biased.")
        loader_iter = self.loader_iter
        while num_data < data_per_loop:
            try:
                images, targets = next(loader_iter)
            except:
                loader_iter = iter(real_loader)
                self.loader_iter = loader_iter
                images, targets = next(loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[0]
            if num_data + batch_size > data_per_loop:
                batch_size = data_per_loop - num_data
                images = images[:batch_size]
                targets = targets[:batch_size]
            num_data += batch_size
            meta_out = forwardloss(backbone, images, targets)
            meta_loss = meta_out[0]
            weight = float(batch_size)/float(data_per_loop)
            metabackward(meta_loss=meta_loss, 
                         weight=weight,
                         dLdw_groups=dLdw_groups_,
                         params=params_,
                         group_startends=group_startends_,
                         _backward_handle=backward_handle_)
            meta_loss_item += meta_loss.item()*weight
        return meta_loss_item

        
    
    # def step(self,
    #          num_forward:int,
    #          num_backward:int,
    #          meta_params:List[Tensor],
    #          batch_kwargs:dict=None,
    #          meta_loss_kwargs:dict=None):
    #     from models.utils import get_model
    #     from utils import get_optimizer
    #     from BPTT.diff_optim import DiffOptimizer
    #     from BPTT.utils import get_diff_opt
    #     backbone = get_model(**self.inner_model_args)
    #     backbone.to(self.device)
    #     opt = get_optimizer(backbone.parameters(), **self.inner_opt_args)
    #     diffopt = get_diff_opt(backbone, opt, True, self._paidx_grpidx_map)
    #     self._paidx_grpidx_map = diffopt.paidx_grpidx_map
    #     diffopt.forward_loop(self.forward_function,
    #                          num_forward,
    #                          num_backward,
    #                          batch_kwargs)
    #     meta_loss_item = self.meta_loss_and_backward(backbone=backbone, **meta_loss_kwargs)
    #     diffopt.backward_loop(num_backward, meta_params)
    #     return meta_loss_item

        
             
        