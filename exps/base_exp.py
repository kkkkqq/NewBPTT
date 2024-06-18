from type_ import *
from synset.base_synset import BaseImageSynset
from synset.imagelab_synset import ImageLabSynSet
from synset.synset_loader import SynSetLoader
from inner_loop.baseloop import InnerLoop
from modules.basemodule import BaseModule
from modules.utils import get_module
from models.utils import get_model
from utils import get_optimizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
import os
import numpy as np
import random
from dataset.utils import get_dataset
from ema_opt.ema_opt import ClipEMAOptimizer



class BaseExperiment():

    def __init__(self, config:dict):
        self.config:dict = config
        self.exp_config = self.parse_exp_config(config)
        self.dataset_config = self.parse_dataset_config(config)
        self.synset_config = self.parse_synset_config(config)
        self.eval_config = self.parse_eval_config(config)
        self.loop_config = self.parse_loop_config(config)
        # experiment settings
        
    
    def parse_exp_config(self, config):
        from exps.utils import seed_everything
        exp_config:dict = copy.deepcopy(config['exp_config'])
        self.seed:int = exp_config['seed']
        self.project:str = exp_config['project']
        self.exp_name:str = exp_config['exp_name'] + 'seed' + str(self.seed)
        self.save_dir:str = './results/'+self.exp_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        seed_everything(self.seed)
        self.use_wandb:bool = exp_config['use_wandb']
        self.wandb_api_key:str = exp_config['wandb_api_key']
        self.num_steps:int = exp_config['num_steps']
        self.device = torch.device(exp_config['device'])
        return exp_config
    
    def parse_dataset_config(self, config)->dict:
        from copy import deepcopy
        dataset_config = deepcopy(config['dataset_config'])
        self.dataset = get_dataset(**dataset_config)
        self.dataset_aux_args = {'channel': self.dataset.channel,
                                 'num_classes': self.dataset.num_classes,
                                 'image_size': self.dataset.image_size}
        return dataset_config
    
    def parse_synset_config(self, config)->dict:
        from copy import deepcopy
        synset_config = deepcopy(config['synset_config'])
        synset_args:dict = synset_config['synset_args']
        if 'channel' not in synset_args:
                synset_args.update(self.dataset_aux_args)
        synset_args['zca'] = self.dataset.zca_trans
        synset_args['device'] = self.device
        synset_opt_args = synset_config['synset_opt_args']
        self.ema_grad_clip = synset_opt_args['ema_grad_clip']
        self.ema_coef = synset_opt_args['ema_coef']
        self.synset_opt_args = synset_opt_args
        self.synset_args = synset_args
        self.synset:ImageLabSynSet = None
        return synset_config
    
    def parse_eval_config(self, config)->dict:
        from copy import deepcopy
        eval_config = deepcopy(config['eval_config'])
        self.eval_interval:int = eval_config['eval_interval']
        self.num_eval:int = eval_config['num_eval']
        self.eval_models_dict:dict = eval_config['eval_models']
        for args in self.eval_models_dict.values():
            if 'channel' not in args['model_args']:
                args['model_args'].update(self.dataset_aux_args)
        self.eval_steps:int = eval_config['eval_steps']
        self.eval_train_module:BaseModule = get_module(**eval_config['eval_train_module_args'])
        self.eval_test_module:BaseModule = get_module(**eval_config['eval_test_module_args'])
        eval_batchsize = eval_config['eval_batchsize']
        self.test_loader = DataLoader(self.dataset.dst_test, eval_batchsize, pin_memory=True, num_workers=4)
        self.upload_visualize:bool = eval_config['upload_visualize']
        self.upload_visualize_interval:int = eval_config['upload_visualize_interval']
        self.save_visualize:bool = eval_config['save_visualize']
        self.save_visualize_interval:bool = eval_config['save_visualize_interval']
        return eval_config

    def parse_loop_config(self, config)->dict:
        from copy import deepcopy
        loop_config:dict = deepcopy(self.config['loop_config'])
        self.bptt_type:str = loop_config['bptt_type']
        self.num_forward:int = loop_config['num_forward']
        self.num_backward:int = loop_config['num_backward']
        inner_loop_config:dict = loop_config['inner_loop_config']
        inner_loop_type:str = inner_loop_config['inner_loop_type']
        inner_loop_args = inner_loop_config['inner_loop_args']
        if 'channel' not in inner_loop_args['inner_model_args']:
            inner_loop_args['inner_model_args'].update(self.dataset_aux_args)
        if 'inner_batch_size' in inner_loop_args:
            if inner_loop_args['inner_batch_size'] is None:
                inner_loop_args['inner_batch_size'] = self.synset.num_items
        else:
            inner_loop_args['inner_batch_size'] = self.synset.num_items
        if 'device' in inner_loop_args:
            if inner_loop_args['device'] is None:
                inner_loop_args['device'] = self.device
        else:
            inner_loop_args['device'] = self.device
        self.inner_loop_args = inner_loop_args
        self.inner_loop_type = inner_loop_type
        self.inner_loop = None
        inner_loop_args['batch_function'] = self.synset.batch
        return loop_config
    
    def init_wandb(self):
        from wandb import login as wandblogin
        from wandb import init as wandbinit
        if self.use_wandb:
            wandblogin(key=self.wandb_api_key)
            wandbinit(project=self.project,
                      reinit=True,
                      name=self.exp_name,
                      config=self.config)
    
    def prepare_synset_and_opts(self):
        synset = self.synset
        synset.train()
        synset_opt_args = self.synset_opt_args
        trainables:dict = synset.trainables
        synset_opts = dict()
        for key, val in trainables.items():
            if key not in synset_opt_args:
                raise AssertionError("{} should be in synset_opt_args!".format(key))
            opt = get_optimizer(val, **synset_opt_args[key])
            if self.ema_grad_clip:
                opt = ClipEMAOptimizer(opt, self.ema_coef)
            synset_opts[key] = opt
        self.synset_opts = synset_opts
        return synset, synset_opts
    
    def eval_synset(self, step, synset, use_wandb):
        from wandb import log
        from exps.utils import evaluate_synset
        metrics = evaluate_synset(self.eval_models_dict, 
                                  self.num_eval, 
                                  self.eval_steps,
                                  self.test_loader,
                                  synset,
                                  self.eval_train_module,
                                  self.eval_test_module,
                                  self.device
                                  )
        if use_wandb:
            log(metrics, step=step)
    
    def upload_visualize_save(self, 
                              step, 
                              synset:ImageLabSynSet, 
                              upload_vis:bool,
                              save_vis:bool
                              ):
        from wandb import log, Image, Histogram
        from torchvision.utils import save_image
        from exps.utils import save_synset
        if upload_vis or save_vis:
            disp_imgs = synset.images_on_display(None).to('cpu')
            imgs_sample = synset.detached_images_sample(None).to('cpu')
        if upload_vis:
            log({'Images': Image(disp_imgs)}, step=step)
            log({'Pixels': Histogram(imgs_sample)}, step=step)
        if save_vis:
            # disp_imgs.div_(torch.max(torch.abs(disp_imgs))*2).add_(0.5)
            save_image(imgs_sample, self.save_dir+'/'+str(step)+'.jpg', nrow = synset.num_classes)
            save_synset(synset, self.save_dir+'/current_synset.pt')

    def _batch_kwargs(self):
        return dict()
    
    def _meta_loss_kwargs(self):
        return dict()

    def one_loop(self,
             innerloop:InnerLoop,
             synset:ImageLabSynSet,
             synset_opts:Dict[str,Union[Optimizer, ClipEMAOptimizer]],
             bptt_type:str, 
             num_forward:int, 
             num_backward:int,
             step:int,
             use_wandb:bool,
             batch_kwargs=dict(),
             meta_loss_kwargs=dict()
             ):
        from wandb import log
        from numpy.random import randint
        for opt in synset_opts.values():
            opt.zero_grad()
        if bptt_type in ['ratbptt', 'rat_bptt']:
            num_forward = randint(num_backward, num_forward)    
        meta_loss_item = innerloop.loop(num_forward, 
                                        num_backward,
                                        synset.flat_trainables,
                                        batch_kwargs,
                                        meta_loss_kwargs
                                        )
        for opt in synset_opts.values():
            opt.step()
        
        if use_wandb and step%10==0:
            metric = {'meta_loss': meta_loss_item}
            for key, val in synset_opts.items():
                if isinstance(val, ClipEMAOptimizer):
                    grad_norms = val.grad_norms
                    emas = val.ema_vals
                    for grp_idx, (grad_norm, ema) in enumerate(zip(grad_norms, emas)):
                        metric['_'.join([key, 'gradnorm', str(grp_idx)])] = grad_norm
                        metric['_'.join([key, 'ema_val', str(grp_idx)])] = ema
            log(metric, step=step)
        print('meta loss at step {}: {}'.format(step, meta_loss_item))
        return None
        
    def run_exp(self):
        from time import time
        self.init_wandb()
        #prepare variables used in the loop
        inner_loop = self.inner_loop
        bptt_type =self.bptt_type
        num_forward = self.num_forward
        num_backward = self.num_backward
        synset, opts = self.prepare_synset_and_opts()
        use_wandb = self.use_wandb
        upload_vis = use_wandb and self.upload_visualize
        upload_visualize_interval_ = self.upload_visualize_interval
        save_vis = self.save_visualize
        save_visualize_interval_ = self.save_visualize_interval
        eval_interval = self.eval_interval
        eval_synset = self.eval_synset
        upload_visualize_save = self.upload_visualize_save
        one_loop = self.one_loop
        batch_kwargs = self._batch_kwargs()
        meta_loss_kwargs = self._meta_loss_kwargs()
        #looping
        for it in range(self.num_steps+1):
            synset.shuffle()
            tm = time()
            one_loop(inner_loop,
                    synset,
                    opts,
                    bptt_type,
                    num_forward,
                    num_backward,
                    it,
                    use_wandb,
                    batch_kwargs,
                    meta_loss_kwargs
                    )
            tm = time()-tm
            print('one loop took {}s.'.format(tm))
            if (it+1)%eval_interval==0:
                eval_synset(it, synset, use_wandb)
            upload_vis_ = upload_vis and (it+1)%upload_visualize_interval_==0
            save_vis_ = save_vis and (it+1)%save_visualize_interval_==0
            upload_visualize_save(it, synset, upload_vis_, save_vis_)

              

            
    



        
        
        
            