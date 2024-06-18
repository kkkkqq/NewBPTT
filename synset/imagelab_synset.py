from synset.base_synset import BaseImageSynset
from type_ import * 
from kornia.enhance import ZCAWhitening as ZCA
from torch.utils.data import DataLoader
import torch.nn.functional as F
from augment.augment import DiffAug
import numpy as np

class ImageLabSynSet(BaseImageSynset):

    def __init__(self,
                 channel:int,
                 num_classes:int,
                 image_size:Tuple[int,int],
                 ipc:int,
                 zca:ZCA=None,
                 device='cuda',
                 train_images:bool=True,
                 train_targets:bool=True,
                 init_type:str='noise_normal',
                 real_loader:DataLoader=None,
                 augment_args:dict = None):
        self.ipc = ipc
        self.zca = zca
        super().__init__(channel,
                         num_classes,
                         image_size,
                         ipc*num_classes,
                         device)
        self.train_images = train_images
        self.train_targets = train_targets
        self.init_type = init_type
        self.images:Tensor = torch.zeros((self.num_items, self.channel, *self.image_size)).to(self.device)
        self.labels:Tensor = torch.repeat_interleave(torch.arange(self.num_classes), self.ipc, dim=0).to('cpu')#labels are not targets, it is always on cpu.
        self._labels:Tensor=self.labels.to(self.device)
        self.targets:Tensor = F.one_hot(self.labels, num_classes).to(torch.float).to(self.device)
        self.flat_trainables = []
        if self.train_images:
            self.trainables['images'] = [self.images]
            self.flat_trainables.append(self.images)
        if self.train_targets:
            self.trainables['targets'] = [self.targets]
            self.flat_trainables.append(self.targets)
        self.init_type = init_type
        init = self.init_type.lower().split('_')
        self.real_loader = real_loader
        if 'noise' in init:
            if 'normal' in init:
                self.noise_init_images(self.images, True)
            else:
                self.noise_init_images(self.images,normalize=False)
        elif 'real' in init:
            if real_loader is None:
                raise AssertionError("chose real init, but real loader is None!")
            else:
                real_samples = torch.cat(self.sample_real_images(self.labels, real_loader), dim=0)
                self.images[:] = real_samples[:]
        else:
            raise ValueError("unrecognized initialization type {}".format(self.init_type))

        self.augment_args = augment_args
        if self.augment_args is None:
            self.augment = DiffAug(strategy='')
        else:
            self.augment = DiffAug(**self.augment_args)

        self.seed_shift = np.random.randint(10000)

    def __getitem__(self, idxes):
        images, targets = self.images[idxes], self.targets[idxes]
        return images, targets
    
    def set_trainables(self):
        self.flat_trainables = []
        if self.train_images:
            self.trainables['images'] = [self.images]
            self.flat_trainables.append(self.images)
        if self.train_targets:
            self.trainables['targets'] = [self.targets]
            self.flat_trainables.append(self.targets)
        return None
    
    def to(self, device):
        self._labels = self._labels.to(device)
        self.images = self.images.to(device)
        self.targets = self.targets.to(device)
        self.set_trainables()
        return None
    
    def batch(self, batch_idx:int, batch_size:int, with_augment:bool=True, reproducible:bool=True):
        from torch.nn.functional import softmax
        sampler = self.sampler
        images = self.images
        targets = self.targets
        if batch_size<self.num_items:
            idxes = sampler.sample_idxes(batch_idx, batch_size)
            imgs = images[idxes]
            tgts = targets[idxes]
        else:
            imgs = images
            tgts = targets
        if with_augment:
            if reproducible:
                seed = self.seed_shift + batch_idx * batch_size
            else:
                seed = -1
            imgs = self.augment(imgs, seed=seed)
        tgts = softmax(tgts, dim=1)
        return imgs, tgts
    
    def shuffle(self):
        super().shuffle()
        self.seed_shift = np.random.randint(10000)
        return None
    
    def detached_images_sample(self, ipc:int=None):
        if ipc is None:
            return self.images.detach().clone()
        else:
            imgs = []
            for cls in range(self.num_classes):
                start_idx = cls*self.ipc
                end_idx = start_idx + ipc
                chunk = self.images[start_idx:end_idx].detach().clone()
                if len(chunk.shape)<4:
                    chunk.unsqueeze_(0)
                imgs.append(chunk)
            images = torch.cat(imgs, dim=0)
            return images
    
    def images_on_display(self, ipc=None, clip_val=None, upsample_factor=None):
        with torch.no_grad():
            images = self.detached_images_sample(ipc)
            if self.zca is not None:
                images = self.zca_inverse_images(images, self.zca)
            if clip_val is not None:
                images = self.clip_images(images, clip_val)
            if upsample_factor is not None:
                images = self.upsample_images(images, upsample_factor)
            images = self.make_grid_images(images, self.num_classes)
        return images
            
            
        
