import torch
from type_ import *
from torch.utils.data import DataLoader
from kornia.enhance import ZCAWhitening as ZCA
from torchvision.utils import make_grid
import numpy as np
from utils import get_optimizer

class IdxSampler():

    def __init__(self, num_items, variety:int=100) -> None:
        self.num_items = num_items
        self.variety = variety
        self.idxes = [np.random.permutation(num_items) for _ in range(variety)]
        return None
    
    def shuffle(self):
        from numpy.random import permutation
        self.idxes = [permutation(self.num_items) for _ in range(self.variety)]
    
    def sample_idxes(self, batch_idx:int, batch_size:int):
        batch_per_line = self.num_items//batch_size
        line_idx = (batch_idx // batch_per_line)//self.variety
        start_idx = (batch_idx%batch_per_line)*batch_size
        end_idx = start_idx + batch_size
        idxes = self.idxes[line_idx][start_idx:end_idx]
        return idxes

class BaseSynSet():

    def __init__(self, num_items:int, device='cuda'):
        """
        Baseclass for synset.
        """
        self.trainables:Dict[str,List[Tensor]] = dict()
        self.flat_trainables:List[Tensor] = []
        self.num_items:int = num_items
        self.device = torch.device(device)
        self.sampler = IdxSampler(self.num_items)

    def __getitem__(self, idxes):
        raise NotImplementedError

    def __len__(self):
        return self.num_items
    
    def to(self, device):
        raise NotImplementedError
    
    def batch(self, batch_idx:int, batch_size:int, class_idx:int=None, tracked:bool=True):
        raise NotImplementedError

    def shuffle(self):
        self.sampler.shuffle()
        return None
    
    def eval(self):
        for itm in self.flat_trainables:
            itm.requires_grad_(False)
        return None
    
    def train(self):
        for itm in self.flat_trainables:
            itm.requires_grad_(True)
        return None
    
class BaseImageSynset(BaseSynSet):

    def __init__(self,
                 channel:int,
                 num_classes:int,
                 image_size:Tuple[int,int],
                 num_items:int,
                 device='cuda'):
        super().__init__(num_items, device)
        self.channel = channel
        self.num_classes = num_classes
        self.image_size = image_size
    
    def detached_images_sample(self):
        raise NotImplementedError

    @staticmethod
    def zca_inverse_images(imgs:Tensor, zca_trans:ZCA):
        with torch.no_grad():
            imgs = zca_trans.inverse_transform(imgs)
        return imgs
    
    @staticmethod
    def upsample_images(imgs:Tensor, repeats:int):
        from torch import no_grad, repeat_interleave
        with no_grad():
            imgs = repeat_interleave(imgs, repeats, 2)
            imgs = repeat_interleave(imgs, repeats, 3)
        return imgs
    
    @staticmethod
    def make_grid_images(imgs:Tensor, 
                         nrow:int=10, 
                         padding:int=2,
                         normalize:bool=False, 
                         value_range:Tuple[int,int]=None,
                         scale_each:bool=False,
                         pad_value:float=0,
                         **kwargs):
        from torch import no_grad
        from torchvision.utils import make_grid
        with no_grad():
            grids=make_grid(imgs,
                            nrow,
                            padding,
                            normalize,
                            value_range,
                            scale_each,
                            pad_value,
                            **kwargs)
        return grids
    
    @staticmethod
    def clip_images(imgs:Tensor,
                    clip_val:float):
        with torch.no_grad():
            std = torch.std(imgs)
            mean = torch.mean(imgs)
            imgs = torch.clip(imgs.detach().clone(), min = mean-clip_val*std, max = mean+clip_val*std)
        return imgs
    
    @staticmethod
    def noise_init_images(images:Tensor, normalize:bool=True):
        with torch.no_grad():
            images[:] = torch.randn_like(images)
            if normalize:
                images_norm = torch.norm(images, p=2, dim=(1,2,3), keepdim=True)
                images.div_(images_norm)
        return None
    
    @staticmethod
    def sample_real_images(labels:Tensor,
                           real_loader:DataLoader):
        """
        randomly sample images from real_loader with labels 
        determined by `labels`
        """
        from torch import no_grad, argmax
        from torch import max as torchmax
        from torch import sum as torchsum
        with no_grad():
            if len(labels.shape)>1:
                labels = argmax(labels, dim=1)
            maxclass = int(torchmax(labels).item())
            cls_imgs_lst = [[] for _ in range(maxclass)]
            cls_num_lst = [0 for _ in range(maxclass)]
            cls_maxnum_lst = [int(torchsum(labels==lab).item()) for lab in range(maxclass)]
            
            for batch in real_loader:
                imgs = batch[0]
                labs = batch[1]
                if len(labs.shape)>1:
                    labs = torch.argmax(labs, dim=1)
                for cat in range(maxclass):
                    idxes = labs==cat
                    num_data = int(torchsum(idxes))
                    cls_imgs = imgs[idxes]
                    if len(cls_imgs.shape)<4:
                        cls_imgs.unsqueeze_(0)
                    cls_imgs_lst[cat].append(cls_imgs)
                    cls_num_lst[cat] += num_data
                if all([num>=maxnum for num, maxnum in zip(cls_num_lst, cls_maxnum_lst)]):
                    break
            cls_imgs_lst = [torch.cat(eles, dim=0) for eles in cls_imgs_lst]
        return cls_imgs_lst