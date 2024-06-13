from exps.imglbrclf_exp import ImgLbrClfExp
from synset.imagelabelertarget_synset import ImageLabelerTargetSynSet
from inner_loop.clfloop import CLFInnerLoop
from torch.utils.data import DataLoader

class ImgLbrTgtClfExp(ImgLbrClfExp):

    def __init__(self, config:dict):
        super().__init__(config)
    
    def parse_synset_config(self, config) -> dict:
        synset_config = super().parse_synset_config(config)
        synset_args = synset_config['synset_args']
        if 'real' in synset_args['init_type']:
            real_loader = DataLoader(self.dataset.dst_train,batch_size=256)
            synset_args['real_loader'] = real_loader
        self.synset = ImageLabelerTargetSynSet(**synset_config['synset_args'])
        return synset_config