import os
from BPTT.diff_adam import DiffOptimizer
from type_ import *
from modules.utils import get_module
from models.utils import get_model
from utils import get_optimizer
from copy import deepcopy
from modules.basemodule import BaseModule
from dataset.baseset import ImageDataSet
from inner_loop.baseloop import InnerLoop
from tqdm import tqdm

class MTTLoop(InnerLoop):
    def __init__(self,
                 batch_function:Callable,
                 inner_module_args:dict,
                 inner_model_args:dict,
                 inner_opt_args:dict,
                 inner_batch_size:int,
                 expert_model_args:dict,
                 expert_opt_args:dict,
                 buffer_folder:str,
                 num_exp:int,
                 max_start_epoch:int,
                 expert_epoch_per_step:int,
                 train_lr:bool,
                 lr_lr:float,
                 device='cuda'):
        super().__init__(batch_function,
                         inner_module_args,
                         inner_model_args,
                         inner_opt_args,
                         inner_batch_size,
                         device)
        self.train_lr = train_lr
        self.lr_lr = lr_lr
        self.buffer_folder = buffer_folder
        self.num_exp = num_exp
        self.max_start_epoch = max_start_epoch
        self.expert_epoch_per_step = expert_epoch_per_step
        self.buffer:List[List[Tuple[Module, dict]]] = []#List[List[(model, opt_state_dict)]]
        self.expert_model_args = expert_model_args
        self.expert_opt_args = expert_opt_args
        template_model = get_model(**expert_model_args)
        template_model.to(self.device)
        print('reading buffers:')
        for exp_idx in tqdm(range(num_exp)):
            exp_buffer = []
            for ep_idx in range(max_start_epoch+expert_epoch_per_step+1):
                buffername = ''.join(['exp', str(exp_idx), '_', 'epoch',str(ep_idx), '.pt'])
                buffername = os.path.join(buffer_folder, buffername)
                stdt = torch.load(buffername)
                model_state = stdt['model']
                opt_stdt = stdt['optimizer']
                template_model.load_state_dict(model_state)
                copymodel = deepcopy(template_model)
                exp_buffer.append((copymodel, opt_stdt))
            self.buffer.append(exp_buffer)
        self.inner_lr = self.inner_opt_args['lr']
            
                



        
    
    def meta_loss_and_backward(self, 
                               backbone:Module, 
                               flat_init_params:Tensor, 
                               flat_end_params:Tensor, 
                               **meta_loss_kwargs)->float:
        from torch import cat
        from torch import sum as torchsum
        flat_params = cat([ele.flatten() for ele in backbone.parameters()])
        meta_loss = torchsum((flat_params-flat_end_params).pow(2)).div(torchsum((flat_init_params-flat_end_params).pow(2)))
        self.diff_opt.meta_backward(meta_loss)
        return meta_loss.item()

        
    
    def loop(self,
             num_forward:int,
             num_backward:int,
             meta_params:List[Tensor],
             batch_kwargs:dict=dict(),
             meta_loss_kwargs:dict=dict())->float:
        from utils import get_optimizer
        from BPTT.diff_optim import DiffOptimizer
        from BPTT.utils import get_diff_opt
        from numpy.random import randint
        from torch import cat
        from copy import deepcopy
        exp_idx = randint(0, self.num_exp)
        start_idx = randint(0, self.max_start_epoch)
        end_idx = start_idx + self.expert_epoch_per_step
        buf_mod, buf_opt_stdt = self.buffer[exp_idx][start_idx]
        backbone = deepcopy(buf_mod)
        backbone.to(self.device)
        opt = get_optimizer(backbone.parameters(), **self.inner_opt_args)
        opt.load_state_dict(buf_opt_stdt)
        inner_lr = self.inner_lr
        for group in opt.param_groups:
            group['lr'] = inner_lr
            print('inner lr is currently {}'.format(inner_lr))
        end_mod, _ = self.buffer[exp_idx][end_idx]
        end_mod.to(self.device)
        with torch.no_grad():
            flat_init_params = cat([ele.flatten() for ele in backbone.parameters()])
            flat_end_params = cat([ele.flatten() for ele in end_mod.parameters()]).to(self.device)
        diffopt = get_diff_opt(backbone, opt, True, self._attpa2modpa_idxes)
        self._attpa2modpa_idxes = diffopt.attpa2modpa_idxes
        self.backbone = backbone
        self.diff_opt = diffopt
        diffopt.forward_loop(self.forward_function,
                             num_forward,
                             num_backward,
                             batch_kwargs)
        meta_loss_item = self.meta_loss_and_backward(backbone=backbone, 
                                                     flat_init_params=flat_init_params, 
                                                     flat_end_params=flat_end_params, 
                                                     **meta_loss_kwargs)
        diffopt.backward_loop(num_backward, meta_params, self.train_lr)
        if self.train_lr:
            self.inner_lr -= self.lr_lr * diffopt.lr_grad.item()
        return meta_loss_item

        
             
        