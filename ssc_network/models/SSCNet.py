#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Credits: Modified version of PALNet (originally by jieli_cn@163.com, https://github.com/waterljwant/SSC)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from ..utils.projection_layer import Project2Dto3D


# ----------------------------------------------------------------------

# takes only the depth as inputs
class SSCNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SSCNet, self).__init__()
        print("SSCNet")

        # ---- depth
        depth_out = 6
        self.conv2d_depth = nn.Sequential(
            nn.Conv2d(1, depth_out, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        in_ch = depth_out // 2
        self.res_depth = nn.Sequential(
            nn.Conv2d(depth_out, in_ch, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, depth_out, 1, 1, 0),
        )

        self.project_layer = Project2Dto3D(
            240, 144, 240)  # w=240, h=144, d=240

        in_channel_3d = depth_out
        stride = 2
        self.pool1 = nn.Conv3d(in_channel_3d, 8, 4, stride, 3, dilation=2)
        self.reduction2_1 = nn.Conv3d(8, 16, 1, 1, 0, bias=False)
        self.conv2_1 = nn.Sequential(
            nn.Conv3d(8, 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, 3, 1, 1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, 1, 1, 0)
        )

        # ---- reduction
        stride = 2
        self.reduction3_1 = nn.Conv3d(16, 32, 1, stride, 0, bias=False)
        self.conv3_1 = nn.Sequential(
            nn.Conv3d(16, 8, 1, stride, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, 3, 1, 1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 32, 1, 1, 0),
        )
        self.expand_depth_feats_channels = nn.Conv3d(
            32, 64, 1, 1, 0, bias=False)

        # -------------1/4

        self.conv3_3 = nn.Sequential(
            nn.Conv3d(64, 32, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, 1, 1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 1, 1, 0),
        )

        self.conv3_5 = nn.Sequential(
            nn.Conv3d(64, 32, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, 1, 2, 2, groups=4),
            nn.Conv3d(32, 32, 3, 1, 2, 2, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 1, 1, 0),
        )

        self.conv3_7 = nn.Sequential(
            nn.Conv3d(64, 32, 1, 1, 0),
            nn.ReLU(inplace=True),









            nn.Conv3d(32, 32, 3, 1, 2, 2, groups=4),
            nn.Conv3d(32, 32, 3, 1, 2, 2, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 1, 1, 0),
        )

        self.conv4_1 = nn.Conv3d(256, 128, 1, 1, 0)

        self.conv4_2 = nn.Conv3d(128, 128, 1, 1, 0)

        # C_NUM = 12, number of classes is 12
        self.fc12 = nn.Conv3d(128, num_classes, 1, 1, 0)

        self.softmax = nn.Softmax(dim=1)  # pytorch 0.3.0

        # ----  weights init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)  # gain=1
        nn.init.normal_(self.conv4_1.weight.data, mean=0, std=0.1)
        nn.init.normal_(self.conv4_2.weight.data, mean=0, std=0.01)
        nn.init.normal_(self.fc12.weight.data, mean=0, std=0.01)

    def forward(self, x_depth, x_tsdf, p): # TODO
        x0_depth = self.conv2d_depth(x_depth)
        x0_depth = F.relu(self.res_depth(x0_depth) + x0_depth, inplace=True)
        x0_depth = self.project_layer(x0_depth, p)

        x1_depth = self.pool1(x0_depth)
        x1_depth = F.relu(x1_depth, inplace=True)

        x2_1_depth = self.reduction2_1(x1_depth)  # (BS, 32L, 120L, 72L, 120L)

        x2_2_depth = self.conv2_1(x1_depth)
        x2_depth = x2_1_depth + x2_2_depth
        x2_depth = F.relu(x2_depth, inplace=True)

        x3_1_depth = self.reduction3_1(x2_depth)  # (BS, 64L, 60L, 36L, 60L)
        x3_2_depth = self.conv3_1(x2_depth)
        x_3_depth = x3_1_depth + x3_2_depth
        x_3_depth = F.relu(x_3_depth, inplace=True)
        x_3 = self.expand_depth_feats_channels(x_3_depth)

        # ---- 1/4
        x_4 = self.conv3_3(x_3) + x_3
        x_4 = F.relu(x_4, inplace=True)

        x_5 = self.conv3_5(x_4) + x_4
        x_5 = F.relu(x_5, inplace=True)

        x_6 = self.conv3_7(x_5) + x_5
        x_6 = F.relu(x_6, inplace=True)

        x_6 = torch.cat((x_3, x_4, x_5, x_6), dim=1)  # channels concatenate

        x_6 = self.conv4_1(x_6)       # (BS, 128L, 60L, 36L, 60L)
        x_6 = F.relu(x_6, inplace=True)

        x_6 = self.conv4_2(x_6)       # (BS, 128L, 60L, 36L, 60L)
        x_6 = F.relu(x_6, inplace=True)
        y = self.fc12(x_6)        # (BS, 12L, 60L, 36L, 60L)

        return y
