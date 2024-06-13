from exps.base_exp import BaseExperiment
from synset.imagelab_synset import ImageLabSynSet
from inner_loop.clfloop import CLFInnerLoop
from torch.utils.data import DataLoader

class ImgTgtClfExp(BaseExperiment):

    def __init__(self, config:dict):
        super().__init__(config)
    
    def parse_synset_config(self, config) -> dict:
        synset_config = super().parse_synset_config(config)
        synset_args = synset_config['synset_args']
        if 'real' in synset_args['init_type']:
            real_loader = DataLoader(self.dataset.dst_train,batch_size=256)
            synset_args['real_loader'] = real_loader
        self.synset = ImageLabSynSet(**synset_config['synset_args'])
        return synset_config
    
    def parse_loop_config(self, config) -> dict:
        loop_config = super().parse_loop_config(config)
        inner_loop_args = self.inner_loop_args
        inner_loop_args['real_dataset'] = self.dataset
        self.inner_loop = CLFInnerLoop(**self.inner_loop_args)
        return loop_config