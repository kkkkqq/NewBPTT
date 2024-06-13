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
                    eval_train_module:BaseModule,
                    eval_test_module:BaseModule,
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
                eval_train_module.epoch(model, opt, synset_loader, False, True)
            train_metric = eval_train_module.epoch(model, opt, synset_loader, True, True)
            for key, val in train_metric.items():
                print('train', key, ':', val)
                if key not in mean_train_metric:
                    mean_train_metric[key] = val/num_eval
                else:
                    mean_train_metric[key] += val/num_eval
            test_metric = eval_test_module.epoch(model, opt, test_loader, True, False)
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
    from copy import deepcopy
    from torch import save
    synsetcopy = deepcopy(synset)
    synsetcopy.to('cpu')
    save(synsetcopy, path)
    return None

def seed_everything(seed:int):
	#  下面两个常规设置了，用来np和random的话要设置 
    from numpy.random import seed as nprandomseed
    from random import seed as randomseed
    from torch.cuda import manual_seed, manual_seed_all
    from torch import manual_seed as torchmanualseed
    nprandomseed(seed) 
    randomseed(seed)
    manual_seed(seed)
    manual_seed_all(seed) # 多GPU训练需要设置这个
    torchmanualseed(seed)