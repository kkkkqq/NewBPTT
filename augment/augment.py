# This code is based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager


class DiffAug():
    def __init__(self,
                 strategy='color_crop_cutout_flip_scale_rotate',
                 batch=False,
                 ratio_cutout=0.5,
                 single=False):
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = ratio_cutout
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

        self.batch = batch

        self.aug = True
        if strategy == '' or strategy.lower() == 'none' or strategy is None:
            self.aug = False
        else:
            self.strategy = []
            self.flip = False
            self.color = False
            self.cutout = False
            for aug in strategy.lower().split('_'):
                if aug == 'flip' and single == False:
                    self.flip = True
                elif aug == 'color' and single == False:
                    self.color = True
                elif aug == 'cutout' and single == False:
                    self.cutout = True
                else:
                    self.strategy.append(aug)

        self.aug_fn = {
            'color': [self.brightness_fn, self.saturation_fn, self.contrast_fn],
            'crop': [self.crop_fn],
            'cutout': [self.cutout_fn],
            'flip': [self.flip_fn],
            'scale': [self.scale_fn],
            'rotate': [self.rotate_fn],
            'translate': [self.translate_fn],
        }

    def __call__(self, x, single_aug=True, seed=-1):
        from numpy.random import randint
        if not self.aug:
            return x
        else:
            with self.set_seed(seed):
                if self.flip:
                    x = self.flip_fn(x, self.batch)
                if self.color:
                    for f in self.aug_fn['color']:
                        x = f(x, self.batch)
                if len(self.strategy) > 0:
                    if single_aug:
                        # single
                        idx = randint(len(self.strategy))
                        p = self.strategy[idx]
                        for f in self.aug_fn[p]:
                            x = f(x, self.batch)
                    else:
                        # multiple
                        for p in self.strategy:
                            for f in self.aug_fn[p]:
                                x = f(x, self.batch)
                if self.cutout:
                    x = self.cutout_fn(x, self.batch)

            x = x.contiguous()
            return x

    @contextmanager
    def set_seed(self, seed):
        from numpy.random import get_state, set_state
        from numpy.random import seed as npseed

        if seed > 0:
            state = get_state()
            npseed(seed)
        yield
        if seed > 0:
            set_state(state)


    def scale_fn(self, x, batch=True):
        from numpy.random import uniform
        from torch import tensor
        from torch import float as torchfloat
        from torch.nn.functional import affine_grid, grid_sample
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.ratio_scale

        if batch:
            sx = uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            theta = [[sx, 0, 0], [0, sy, 0]]
            theta = tensor(theta, dtype=torchfloat, device=x.device)
            theta = theta.expand(x.shape[0], 2, 3)
        else:
            sx = uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])]
            theta = tensor(theta, dtype=torchfloat, device=x.device)

        grid = affine_grid(theta, x.shape, align_corners=False)
        x = grid_sample(x, grid, align_corners=False)
        return x

    def rotate_fn(self, x, batch=True):
        from numpy.random import uniform
        from numpy import pi, cos, sin
        from torch import tensor
        from torch import float as torchfloat
        from torch.nn.functional import affine_grid, grid_sample
        # [-180, 180], 90: anticlockwise 90 degree
        ratio = self.ratio_rotate

        if batch:
            theta = (uniform() - 0.5) * 2 * ratio / 180 * float(pi)
            theta = [[cos(theta), sin(-theta), 0], [sin(theta), cos(theta), 0]]
            theta = tensor(theta, dtype=torchfloat, device=x.device)
            theta = theta.expand(x.shape[0], 2, 3)
        else:
            theta = (uniform(size=x.shape[0]) - 0.5) * 2 * ratio / 180 * float(pi)
            theta = [[[cos(theta[i]), sin(-theta[i]), 0],
                      [sin(theta[i]), cos(theta[i]), 0]] for i in range(x.shape[0])]
            theta = tensor(theta, dtype=torchfloat, device=x.device)

        grid = affine_grid(theta, x.shape, align_corners=False)
        x = grid_sample(x, grid, align_corners=False)
        return x

    def flip_fn(self, x, batch=True):
        from torch import from_numpy, where
        from numpy.random import uniform
        prob = self.prob_flip

        if batch:
            coin = uniform()
            if coin < prob:
                return x.flip(3)
            else:
                return x
        else:
            randf = from_numpy(uniform(0, 1, (x.size(0), 1, 1, 1))).to(x.device)
            #randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            return where(randf < prob, x.flip(3), x)

    def brightness_fn(self, x, batch=True):
        from torch import from_numpy
        from numpy.random import uniform
        # mean
        ratio = self.brightness

        if batch:
            randb = uniform()
        else:
            randb = from_numpy(uniform(0, 1, (x.size(0), 1, 1, 1))).to(dtype=x.dtype, device=x.device)
            #randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (randb - 0.5) * ratio
        return x

    def saturation_fn(self, x, batch=True):
        from torch import from_numpy, where
        from numpy.random import uniform
        # channel concentration
        ratio = self.saturation

        x_mean = x.mean(dim=1, keepdim=True)
        if batch:
            rands = uniform()
        else:
            rands = from_numpy(uniform(0, 1, (x.size(0), 1, 1, 1))).to(dtype=x.dtype, device=x.device)
            #rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def contrast_fn(self, x, batch=True):
        from torch import from_numpy
        from numpy.random import uniform
        # spatially concentrating
        ratio = self.contrast

        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        if batch:
            randc = uniform()
        else:
            randc = from_numpy(np.random.uniform(0, 1, (x.size(0), 1, 1, 1))).to(dtype=x.dtype, device=x.device)
            # randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    def translate_fn(self, x, batch=True):
        from numpy.random import randint
        from torch import from_numpy, meshgrid, arange, clamp
        from torch import long as torchlong
        from torch.nn.functional import pad
        ratio = self.ratio_crop_pad

        shift_y = int(x.size(3) * ratio + 0.5)
        if batch:
            translation_y = randint(-shift_y, shift_y + 1)
        else:
            translation_y = from_numpy(randint(-shift_y, shift_y+1, (x.size(0), 1, 1))).to(x.device)
            # translation_y = torch.randint(-shift_y,
            #                               shift_y + 1,
            #                               size=[x.size(0), 1, 1],
            #                               device=x.device)

        grid_batch, grid_x, grid_y = meshgrid(
            arange(x.size(0), dtype=torchlong, device=x.device),
            arange(x.size(2), dtype=torchlong, device=x.device),
            arange(x.size(3), dtype=torchlong, device=x.device),
        )
        grid_y = clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = pad(x, (1, 1))
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def crop_fn(self, x, batch=True):
        from numpy.random import randint
        from torch import from_numpy, meshgrid, clamp, arange
        from torch.nn.functional import pad
        from torch import long as torchlong
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad

        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        if batch:
            translation_x = randint(-shift_x, shift_x + 1)
            translation_y = randint(-shift_y, shift_y + 1)
        else:
            translation_x = from_numpy(randint(-shift_x, shift_x+1, (x.size(0), 1, 1))).to(x.device)
            # translation_x = torch.randint(-shift_x,
            #                               shift_x + 1,
            #                               size=[x.size(0), 1, 1],
            #                               device=x.device)

            translation_y = from_numpy(randint(-shift_y, shift_y+1, (x.size(0), 1, 1))).to(x.device)
            # translation_y = torch.randint(-shift_y,
            #                               shift_y + 1,
            #                               size=[x.size(0), 1, 1],
            #                               device=x.device)

        grid_batch, grid_x, grid_y = meshgrid(
            arange(x.size(0), dtype=torchlong, device=x.device),
            arange(x.size(2), dtype=torchlong, device=x.device),
            arange(x.size(3), dtype=torchlong, device=x.device),
        )
        grid_x = clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = pad(x, (1, 1, 1, 1))
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def cutout_fn(self, x, batch=True):
        from numpy.random import randint
        from torch import from_numpy, meshgrid, clamp, arange, ones
        from torch.nn.functional import pad
        from torch import long as torchlong
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = randint(0, x.size(2) + (1 - cutout_size[0] % 2))
            offset_y = randint(0, x.size(3) + (1 - cutout_size[1] % 2))
        else:
            offset_x = from_numpy(randint(0,
                                          x.size(2) + (1 - cutout_size[0] % 2),
                                          size=(x.size(0), 1, 1),
                                          )).to(x.device)
            # offset_x = torch.randint(0,
            #                          x.size(2) + (1 - cutout_size[0] % 2),
            #                          size=[x.size(0), 1, 1],
            #                          device=x.device)

            offset_y = from_numpy(randint(0,
                                        x.size(3) + (1 - cutout_size[1] % 2),
                                        size=[x.size(0), 1, 1],
                                        )).to(x.device)
            # offset_y = torch.randint(0,
            #                          x.size(3) + (1 - cutout_size[1] % 2),
            #                          size=[x.size(0), 1, 1],
            #                          device=x.device)

        grid_batch, grid_x, grid_y = meshgrid(
            arange(x.size(0), dtype=torchlong, device=x.device),
            arange(cutout_size[0], dtype=torchlong, device=x.device),
            arange(cutout_size[1], dtype=torchlong, device=x.device),
        )
        grid_x = clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x

    def cutout_inv_fn(self, x, batch=True):
        from numpy.random import randint
        from torch import from_numpy, meshgrid, clamp, arange, zeros
        from torch import long as torchlong
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = randint(0, x.size(2) - cutout_size[0])
            offset_y = randint(0, x.size(3) - cutout_size[1])
        else:
            offset_x = from_numpy(randint(0, 
                                        x.size(2) - cutout_size[0],
                                        size=[x.size(0), 1, 1],)).to(x.device)
            # offset_x = torch.randint(0,
            #                          x.size(2) - cutout_size[0],
            #                          size=[x.size(0), 1, 1],
            #                          device=x.device)
            offset_y = from_numpy(randint(0, 
                                        x.size(3) - cutout_size[1],
                                        size=[x.size(0), 1, 1],)).to(x.device)
            # offset_y = torch.randint(0,
            #                          x.size(3) - cutout_size[1],
            #                          size=[x.size(0), 1, 1],
            #                          device=x.device)

        grid_batch, grid_x, grid_y = meshgrid(
            arange(x.size(0), dtype=torchlong, device=x.device),
            arange(cutout_size[0], dtype=torchlong, device=x.device),
            arange(cutout_size[1], dtype=torchlong, device=x.device),
        )
        grid_x = clamp(grid_x + offset_x, min=0, max=x.size(2) - 1)
        grid_y = clamp(grid_y + offset_y, min=0, max=x.size(3) - 1)
        mask = zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 1.
        x = x * mask.unsqueeze(1)
        return x
