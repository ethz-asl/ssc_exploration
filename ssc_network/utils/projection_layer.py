#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Project feature tensers of 2D image to 3D space
jieli_cn@163.com
"""

import torch.nn as nn
from torch_scatter import scatter_max


class Project2Dto3D(nn.Module):
    def __init__(self, w=240, h=144, d=240):
        super(Project2Dto3D, self).__init__()
        self.w = w
        self.h = h
        self.d = d

    def forward(self, x2d, idx):
        # bs, c, img_h, img_w = x2d.shape
        bs, c, _, _ = x2d.shape
        src = x2d.view(bs, c, -1)
        idx = idx.view(bs, 1, -1)
        index = idx.expand(-1, c, -1)  # expand to c channels

        x3d = x2d.new_zeros((bs, c, self.w*self.h*self.d))
        x3d, _ = scatter_max(src, index, out=x3d)  # dim_size=240*144*240,

        x3d = x3d.view(bs, c, self.w, self.h, self.d)  # (BS, c, vW, vH, vD)
        x3d = x3d.permute(0, 1, 4, 3, 2)     # (BS, c, vW, vH, vD)--> (BS, c, vD, vH, vW)
        return x3d
