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

def evaluate_synset(eval_models_dict:dict, 
                    num_eval:int,
                    eval_steps:int,
                    test_loader:DataLoader,
                    synset:BaseImageSynset,
                    eval_module:BaseModule,
                    device='cuda'):
    synset_loader = SynSetLoader(synset, synset.num_items, synset.num_items, True)
    metrics = dict()
    for name, args in eval_models_dict.items():
        print('evaluating synset on', name,':')
        model_args = args['model_args']
        opt_args = args['opt_args']
        mean_train_metric = dict()
        mean_test_metric = dict()
        for _ in range(num_eval):
            model = get_model(**model_args)
            model.to(device)
            opt = get_optimizer(model.parameters(), **opt_args)
            for _ in tqdm(range(eval_steps-1)):
                eval_module.epoch(model, opt, synset_loader, False, True)
            train_metric = eval_module.epoch(model, opt, synset_loader, True, True)
            for key, val in train_metric.items():
                print('train', key, ':', val)
                if key not in mean_train_metric:
                    mean_train_metric[key] = val/num_eval
                else:
                    mean_train_metric[key] += val/num_eval
            test_metric = eval_module.epoch(model, opt, test_loader, True, False)
            for key, val in test_metric.items():
                print('test', key, ':', val)
                if key not in mean_test_metric:
                    mean_test_metric[key] = val/num_eval
                else:
                    mean_test_metric[key] += val/num_eval
        for key, val in mean_train_metric.items():
            print('mean train', key, 'for', name, ':', round(val, 4))
        for key, val in mean_test_metric.items():
            print('mean test', key, 'for', name, ':', round(val,4))
        metrics.update({'eval/'+name+'_train_'+ key:val for key,val in mean_train_metric.items()})
        metrics.update({'eval/'+name+'_test_'+ key:val for key,val in mean_test_metric.items()})    
    return metrics       

def save_synset(synset, path):
    synsetcopy = copy.deepcopy(synset)
    synsetcopy.to('cpu')
    torch.save(synsetcopy, path)
    return None

def seed_everything(seed:int):
	#  下面两个常规设置了，用来np和random的话要设置 
    np.random.seed(seed) 
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
    torch.manual_seed(seed)

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
        exp_config:dict = copy.deepcopy(config['exp_config'])
        self.seed:int = exp_config['seed']
        self.project:str = exp_config['project']
        self.exp_name:str = exp_config['exp_name'] + 'seed' + str(self.seed)
        self.save_dir:str = './results/'+self.exp_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        seed_everything(self.seed)
        self.use_wandb:bool = self.exp_config['use_wandb']
        self.wandb_api_key:str = self.exp_config['wandb_api_key']
        self.num_steps:int = self.exp_config['num_steps']
        self.device = torch.device(self.exp_config['device'])
        return exp_config
    
    def parse_dataset_config(self, config)->dict:
        dataset_config = copy.deepcopy(config['dataset_config'])
        self.dataset = get_dataset(**dataset_config)
        self.dataset_aux_args = {'channel': self.dataset.channel,
                                 'num_classes': self.dataset.num_classes,
                                 'image_size': self.dataset.image_size}
        return dataset_config
    
    def parse_synset_config(self, config)->dict:
        synset_config = copy.deepcopy(config['synset_config'])
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
        eval_config = copy.deepcopy(config['eval_config'])
        self.eval_interval:int = eval_config['eval_interval']
        self.num_eval:int = eval_config['num_eval']
        self.eval_models_dict:dict = eval_config['eval_models']
        for args in self.eval_models_dict.values():
            if 'channel' not in args['model_args']:
                args['model_args'].update(self.dataset_aux_args)
        self.eval_steps:int = eval_config['eval_steps']
        self.eval_module:BaseModule = get_module(**eval_config['eval_module_args'])
        eval_batchsize = eval_config['eval_batchsize']
        self.test_loader = DataLoader(self.dataset.dst_test, eval_batchsize, pin_memory=True, num_workers=4)
        self.upload_visualize:bool = eval_config['upload_visualize']
        self.upload_visualize_interval:int = eval_config['upload_visualize_interval']
        self.save_visualize:bool = eval_config['save_visualize']
        self.save_visualize_interval:bool = eval_config['save_visualize_interval']
        return eval_config

    def parse_loop_config(self, config)->dict:
        loop_config:dict = copy.deepcopy(self.config['loop_config'])
        self.bptt_type:str = loop_config['bptt_type']
        self.num_forward:int = loop_config['num_forward']
        self.num_backward:int = loop_config['num_backward']
        inner_loop_config:dict = loop_config['inner_loop_config']
        inner_loop_type:str = inner_loop_config['loop_type']
        inner_loop_args = inner_loop_config['loop_args']
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
        self.inner_loop = None
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
        for key, val in trainables:
            assert key in synset_opt_args
            opt = get_optimizer(val, **synset_opt_args[key])
            if self.ema_grad_clip:
                opt = ClipEMAOptimizer(opt, self.ema_coef)
            synset_opts[key] = opt
        self.synset_opts = synset_opts
        return synset, synset_opts
    
    def eval_synset(self, step, synset):
        from wandb import log
        metrics = evaluate_synset(self.eval_models_dict, 
                                  self.num_eval, 
                                  self.eval_steps,
                                  self.test_loader,
                                  synset,
                                  self.eval_module,
                                  self.device
                                  )
        if self.use_wandb:
            log(metrics, step=step)
    
    def upload_visualize_save(self, step, synset:ImageLabSynSet):
        from wandb import log, Image, Histogram
        from torchvision.utils import save_image
        upload_vis = self.use_wandb and self.upload_visualize and (step+1)%self.upload_visualize_interval==0
        save_vis = self.save_visualize and (step+1)%self.save_visualize_interval==0
        if upload_vis or save_vis:
            disp_imgs = synset.images_on_display(None)
            imgs_sample = synset.detached_images_sample(None)
        if upload_vis:
            log({'Images': Image(disp_imgs)}, step=step)
            log({'Pixels': Histogram(imgs_sample)}, step=step)
        if save_vis:
            # disp_imgs.div_(torch.max(torch.abs(disp_imgs))*2).add_(0.5)
            save_image(imgs_sample, self.save_dir+'/'+str(step)+'.jpg', nrow = synset.num_classes)
            save_synset(synset, self.save_dir+'/current_synset.pt')

    def loop(self,
             innerloop:InnerLoop,
             synset:ImageLabSynSet,
             synset_opts:Dict[str,Union[Optimizer, ClipEMAOptimizer]],
             bptt_type:str, 
             num_forward:int, 
             num_backward:int,
             step:int,
             batch_kwargs:dict=dict(),
             meta_loss_kwargs:dict=dict(),
             ):
        from wandb import log
        for opt in synset_opts.values():
            opt.zero_grad()
        if bptt_type in ['ratbptt', 'rat_bptt']:
            num_forward = np.random.randint(num_backward, num_forward)    
        meta_loss_item = innerloop.loop(num_forward, 
                                        num_backward,
                                        synset.flat_trainables,
                                        batch_kwargs,
                                        meta_loss_kwargs
                                        )
        for opt in synset_opts.values():
            opt.step()
        
        if self.use_wandb:
            metric = {'meta_loss': meta_loss_item}
            for key, val in synset_opts.items():
                if isinstance(val, ClipEMAOptimizer):
                    grad_norms = val.grad_norms
                    emas = val.ema_vals
                    for grp_idx in enumerate(grad_norms):
                        metric['_'.join([key, 'gradnorm', grp_idx])] = grad_norms[grp_idx]
                        metric['_'.join([key, 'ema', grp_idx])] = emas[grp_idx]
            log(metric, step=step)
        
        if step%10==0:
            print('meta loss at step {}: {}'.format(step, meta_loss_item))
        
                

            
    



        
        
        
            