from synset.imagelabeler_synset import ImageLabelerSynSet
from type_ import * 
from kornia.enhance import ZCAWhitening as ZCA
from torch.utils.data import DataLoader
from models.utils import get_model

class ImageLabelerTargetSynSet(ImageLabelerSynSet):

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
        super().__init__(channel,
                         num_classes,
                         image_size,
                         ipc,
                         labeler_args,
                         labeler_path,
                         zca,
                         device,
                         train_images,
                         train_labeler,
                         init_type,
                         real_loader,
                         augment_args)
        return None
        
    def batch(self, batch_idx: int, batch_size: int, with_augment: bool = True, reproducible: bool = True):
        return super().batch(batch_idx, batch_size, with_augment, reproducible, with_label=True)
