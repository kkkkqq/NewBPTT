from torch.nn.modules import Module
from modules.clfmodule import ClassifierModule
from augment.augment import DiffAug
import torch
from torch import Tensor
import torch.nn as nn


class ClassifierLbrTgtModule(ClassifierModule):

    def __init__(self, aug_args:dict=None, label_weight:float=1.0):
        super().__init__(aug_args=aug_args)
        self.label_weight = label_weight
        return None
    
    def forward_loss(self, backbone: Module, images:Tensor, targets:Tensor, labels:Tensor) -> tuple:
        if self._device is None:
            self._device = next(backbone.parameters()).device
        _device = self._device
        criterion = self.criterion
        if images.device != _device:
            images = images.to(_device)
        if targets.device != _device:
            targets = targets.to(_device)
        if labels.device != _device:
            labels = labels.to(_device)
        out = backbone(self.augment(images))
        loss = criterion(out, targets)
        loss = loss + criterion(out, labels)*self.label_weight
        return loss, out, targets
    
    # def post_loss(self, loss:Tensor, out:Tensor, targets:Tensor):
    #     from torch import argmax, eq
    #     from torch import sum as torch_sum
    #     from torch import long as torchlong
    #     batchsize = out.shape[0]
    #     out_argmax = argmax(out, dim=1).to(torchlong)
    #     if len(targets.shape)>1:
    #         targets_argmax = argmax(targets, dim=1).to(torchlong)
    #     else:
    #         targets_argmax = targets
    #     acc = torch_sum(eq(out_argmax, targets_argmax))
    #     return batchsize, {'loss':loss.item()*batchsize, 'acc':acc.item()}

    # def parse_batch(self, batch_out: tuple):
    #     return {'images':batch_out[0], 'targets':batch_out[1]}
        