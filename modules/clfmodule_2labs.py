from torch.nn.modules import Module
from modules.basemodule import BaseModule
from augment.augment import DiffAug
import torch
from torch import Tensor
import torch.nn as nn


class ClassifierModule2Labs(BaseModule):

    def __init__(self, aug_args:dict=None, label_weight:float=1.):
        super().__init__()
        if aug_args is not None:
            self.augment = DiffAug(**aug_args)
        else:
            self.augment = DiffAug(strategy='')
        self.criterion = nn.CrossEntropyLoss()
        self.label_weight = label_weight
        return None
    
    def forward_loss(self, backbone: Module, images:Tensor, targets:Tensor, labels:Tensor) -> tuple:
        if self._device is None:
            self._device = next(backbone.parameters()).device
        if images.device != self._device:
            images = images.to(self._device)
        if targets.device != self._device:
            targets = targets.to(self._device)
        if labels.device != self._device:
            labels = labels.to(self._device)
        out = backbone(self.augment(images))
        loss = self.criterion(out, targets)
        loss = loss + self.criterion(out, labels)*self.label_weight
        return loss, out, targets, labels
    
    def post_loss(self, loss:Tensor, out:Tensor, targets:Tensor, labels:Tensor):
        batchsize = out.shape[0]
        out_argmax = torch.argmax(out, dim=1).to(torch.long)
        if len(targets.shape)>1:
            targets_argmax = torch.argmax(targets, dim=1).to(torch.long)
        else:
            targets_argmax = targets
        acc = torch.sum(torch.eq(out_argmax, targets_argmax))
        return batchsize, {'loss':loss.item()*batchsize, 'acc':acc.item()}

    def parse_batch(self, batch_out: tuple):
        return {'images':batch_out[0], 'targets':batch_out[1], 'labels':batch_out[2]}
        
        