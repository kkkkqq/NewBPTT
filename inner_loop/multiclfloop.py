from BPTT.diff_adam import DiffOptimizer
from type_ import *
from modules.utils import get_module
from modules.basemodule import BaseModule
from dataset.baseset import ImageDataSet
from torch.utils.data import DataLoader
from inner_loop.baseloop import InnerLoop

class MultiCLFInnerLoop(InnerLoop):
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
                 num_meta_loss:int,
                 min_steps:int,
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
        self.num_meta_loss = num_meta_loss
        self.min_steps = min_steps #cannot compute meta loss for steps smaller than it
    # def forward_function(self, step_idx:int, backbone:Module, batch_kwargs:dict):
    #     batch_out = self.batch_function(batch_idx=step_idx, batch_size=self.inner_batch_size, **batch_kwargs)
    #     loss = self.inner_module.forward_loss(backbone=backbone, *batch_out)
    #     return loss
    
    def meta_loss_and_backward(self, 
                               backbone:Module, 
                               divide_by:float=1.,
                               stored_data:List[Tuple[Tensor,Tensor]]=None, 
                               **meta_loss_kwargs)->float:
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
        if stored_data is None:
            store_data=True
            stored_data = []
        else:
            store_data=False
            stored_data_iter = iter(stored_data)
        while num_data < data_per_loop:
            if store_data:
                try:
                    images, targets = next(loader_iter)
                except:
                    loader_iter = iter(real_loader)
                    self.loader_iter = loader_iter
                    images, targets = next(loader_iter)
                stored_data.append((images, targets))
            else:
                images, targets = next(stored_data_iter)
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
            weight /= divide_by #多个时间点的metaloss取平均
            metabackward(meta_loss=meta_loss, 
                         weight=weight,
                         dLdw_groups=dLdw_groups_,
                         params=params_,
                         group_startends=group_startends_,
                         _backward_handle=backward_handle_)
            meta_loss_item += meta_loss.item()*weight
        return meta_loss_item, stored_data

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
        
        
        num_meta_loss = self.num_meta_loss
        stones = [max(self.min_steps, num_forward-num_backward)]
        #stones = [num_forward - num_backward]
        if num_meta_loss > 1:
            stones.extend(self.cand_steps(self.min_steps, 
                                        num_forward, 
                                        num_backward, 
                                        num_meta_loss))
        print('stones: ', stones)
        

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
                             stones[-1],
                             min(num_backward, num_forward-self.min_steps),
                             forward_args_lst,
                             forward_kwargs_lst,
                             )
        stored_data = None
        divide_by = num_meta_loss
        for idx in range(len(stones)-1, 0, -1):
            #print('stone: {}, cur_idx: {}'.format(stones[idx], diffopt.cur_idx))
            backbone = diffopt.model
            meta_loss_item, stored_data = self.meta_loss_and_backward(backbone=backbone, 
                                                                      stored_data=stored_data, 
                                                                      divide_by=divide_by,
                                                                      **meta_loss_kwargs)
            num_loop = stones[idx]-stones[idx-1]
            #print('number of bacward loops: ', num_loop)
            if num_loop == 0:
                break
            
            diffopt.backward_loop(num_loop, meta_params)
        return meta_loss_item
        
    
    
    @staticmethod
    def cand_steps(min_steps, num_forward, num_backward, num_interval):
        from numpy import arange
        from numpy.random import randint
        min_steps = max(min_steps, num_forward-num_backward)
        max_steps = num_forward
        full_interval = max_steps-min_steps
        if full_interval<num_interval:
            raise AssertionError("meta loss can be computed in interval {} to {}, this interval is shorter than {}!".format(min_steps, max_steps, num_interval))
        interval = full_interval//num_interval
        if full_interval%num_interval==0:
            candi_itvs = arange(min_steps, max_steps, interval)
        else:
            candi_itvs = arange(min_steps, min_steps+interval*num_interval, interval)
            candi_itvs[-1] = max_steps-interval
        # print('candi_itvs: ', candi_itvs)
        candi_itvs += randint(0, interval, num_interval)
        candi_itvs = list(candi_itvs)
        return candi_itvs
        