from torch.optim import Optimizer
from type_ import *

class ClipEMAOptimizer():

    def __init__(self,
                 opt:Optimizer,
                 ema_coef:float=0.9):
        self.opt = opt
        self._read_optimizer(opt)
        self.state = opt.state
        self.ema_vals:List[float] = [-1e5 for _ in self.param_groups]
        self.grad_norms:List[float] = [-1e5 for _ in self.param_groups]
        self.ema_coef = ema_coef

    def _read_optimizer(self, opt:Optimizer):
        self.defaults = opt.defaults
        self.param_groups = opt.param_groups
        self.state = opt.state
    
    def add_param_group(self, param_group):
        self.opt.add_param_group(param_group=param_group)
        self.ema_vals.append(-1e5)
        
    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)
        self._read_optimizer(self.opt)

    def state_dict(self):
        self.opt.state_dict()

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, closure:Callable=None):
        from torch import no_grad
        from torch import sum as torchsum
        from torch.nn.utils import clip_grad_norm_
        ema_vals = self.ema_vals
        param_groups = self.param_groups
        for idx, group in enumerate(param_groups):
            shadow = self.ema_vals[idx]
            pas = group['params']
            with no_grad():
                grad_norm = sum([torchsum(ele.grad.pow(2)) for ele in pas]).pow(0.5).item()
            if shadow == -1e5:
                ema_vals[idx] = grad_norm
            else:
                ema_vals[idx] -= (1-self.ema_coef)*(shadow - grad_norm)
            clip_grad_norm_(pas, max_norm=2*ema_vals[idx])
            self.grad_norms[idx] = grad_norm
        self.opt.step(closure)