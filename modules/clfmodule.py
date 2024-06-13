from torch.nn.modules import Module
from modules.basemodule import BaseModule
from augment.augment import DiffAug
import torch
from torch import Tensor
import torch.nn as nn


class ClassifierModule(BaseModule):

    def __init__(self, aug_args:dict=None):
        super().__init__()
        if aug_args is not None:
            self.augment = DiffAug(**aug_args)
        else:
            self.augment = DiffAug(strategy='')
        self.criterion = nn.CrossEntropyLoss()
        return None
    
    def forward_loss(self, backbone: Module, images:Tensor, targets:Tensor) -> tuple:
        from torch.nn.functional import softmax
        if self._device is None:
            self._device = next(backbone.parameters()).device
        _device = self._device
        if images.device != _device:
            images = images.to(_device)
        if targets.device != _device:
            targets = targets.to(_device)
        out = backbone(self.augment(images))
        loss = self.criterion(out, targets)
        return loss, out, targets
    
    def post_loss(self, loss:Tensor, out:Tensor, targets:Tensor):
        from torch import argmax, eq
        from torch import sum as torch_sum
        from torch import long as torchlong
        batchsize = out.shape[0]
        out_argmax = argmax(out, dim=1).to(torchlong)
        if len(targets.shape)>1:
            targets_argmax = argmax(targets, dim=1).to(torchlong)
        else:
            targets_argmax = targets
        acc = torch_sum(eq(out_argmax, targets_argmax))
        return batchsize, {'loss':loss.item()*batchsize, 'acc':acc.item()}

    # def parse_batch(self, batch_out: tuple):
    #     return {'images':batch_out[0], 'targets':batch_out[1]}
        