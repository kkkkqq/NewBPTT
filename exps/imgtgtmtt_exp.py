from exps.imgtgtclf_exp import ImgTgtClfExp
from synset.imagelab_synset import ImageLabSynSet
from torch.utils.data import DataLoader
from inner_loop.mttloop import MTTLoop

class ImgTgtMTTExp(ImgTgtClfExp):

    def __init__(self, config:dict):
        super().__init__(config)
    
    def parse_loop_config(self, config) -> dict:
        loop_config = super().parse_loop_config(config)
        inner_loop_args = self.inner_loop_args
        self.inner_loop = MTTLoop(**inner_loop_args)
        return loop_config