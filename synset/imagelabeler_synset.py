from synset.base_synset import BaseImageSynset
from synset.imagelab_synset import ImageLabSynSet
from type_ import * 
from kornia.enhance import ZCAWhitening as ZCA
from torch.utils.data import DataLoader
import torch.nn.functional as F
from augment.augment import DiffAug
import numpy as np
from models.utils import get_model

class ImageLabelerSynSet(ImageLabSynSet):

    def __init__(self,
                 channel:int,
                 num_classes:int,
                 image_size:Tuple[int,int],
                 ipc:int,
                 labeler_args:dict,
                 labeler_path:str = None,
                 zca:ZCA=None,
                 device='cuda',
                 train_images:bool=True,
                 train_labeler:bool=True,
                 init_type:str='noise_normal',
                 real_loader:DataLoader=None,
                 augment_args:dict = None):
        self.ipc = ipc
        self.zca = zca
        super().__init__(channel,
                         num_classes,
                         image_size,
                         ipc,
                         zca,
                         device,
                         train_images,
                         False,
                         init_type,
                         real_loader,
                         augment_args
        )
        self.train_labeler = train_labeler
        self.labeler_args = labeler_args
        self.labeler:Module = get_model(**labeler_args)
        if labeler_path is not None:
            self.labeler.load_state_dict(torch.load(labeler_path))
        self.labeler.to(self.device)
        self.labeler_path = labeler_path
        self.set_trainables()
    
    def set_trainables(self):
        ImageLabSynSet.set_trainables(self)
        if self.train_labeler:
            self.trainables['labeler'] = list(self.labeler.parameters())
            self.flat_trainables.extend(self.trainables['labeler'])
        return None
    
    def to(self, device):
        self.labeler.to(device)
        self.set_trainables()
        return None
    
    def batch(self, batch_idx:int, batch_size:int, with_augment:bool=True, reproducible:bool=True, with_label:bool=False):
        imgs, lbs = ImageLabSynSet.batch(self, batch_idx, batch_size, with_augment, reproducible)
        tgts = self.labeler(imgs)
        if with_label:
            return imgs, tgts, lbs
        else:
            return imgs, tgts
