#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import copy
import multiprocessing
import os

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import resnet50, Bottleneck
from torchvision.transforms import functional

class MGN(nn.Module):

    def __init__(self, num_classes):
        super(MGN, self).__init__()

        resnet = resnet50()
        # 导入模型参数
        model_params_path = "/home/wangz/baseline/resnet50.pkl"
        net_params = torch.load(model_params_path)
        # 把加载的参数应用到模型中
        resnet.load_state_dict(net_params)

        # backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3[0],  # res_conv4_1
        )

        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        # res_conv5 global
        res_g_conv5 = resnet.layer4
        # res_conv5 part
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        # mgn part-1 global
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        # mgn part-2
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        # mgn part-3
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # global max pooling
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(16, 16))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(16, 16))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(8, 16))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(6, 16),padding = (1,0))

        # Figure 3: Notice that the 1 × 1 convolutions for dimension reduction and fully connected layers for identity
        # prediction in each branch DO NOT share weights with each other.

        # 4. Experiment 4.1 Implementation: Notice that different branches in the network are all initialized with the
        # same pretrained weights of the corresponding layers after the res conv4 1 block.

        # conv1 reduce
        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # fc softmax loss
        self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        self.fc_id_2048_2 = nn.Linear(2048, num_classes)
        self.fc_id_256_1_0 = nn.Linear(256, num_classes)
        self.fc_id_256_1_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_0 = nn.Linear(256, num_classes)
        self.fc_id_256_2_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_2 = nn.Linear(256, num_classes)
        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)
        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)  # z_g^G
        zg_p2 = self.maxpool_zg_p2(p2)  # z_g^P2
        zg_p3 = self.maxpool_zg_p3(p3)  # z_g^P3

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]  # z_p0^P2
        z1_p2 = zp2[:, :, 1:2, :]  # z_p1^P2

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]  # z_p0^P3
        z1_p3 = zp3[:, :, 1:2, :]  # z_p1^P3
        z2_p3 = zp3[:, :, 2:3, :]  # z_p2^P3

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)  # f_g^G, L_triplet^G
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)  # f_g^P2, L_triplet^P2
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)  # f_g^P3, L_triplet^P3
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)  # f_p0^P2
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)  # f_p1^P2
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)  # f_p0^P3
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)  # f_p1^P3
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)  # f_p2^P3

        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))  # L_softmax^G
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P2
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P3
        l0_p2 = self.fc_id_256_1_0(f0_p2)  # L_softmax0^P2
        l1_p2 = self.fc_id_256_1_1(f1_p2)  # L_softmax1^P2
        l0_p3 = self.fc_id_256_2_0(f0_p3)  # L_softmax0^P3
        l1_p3 = self.fc_id_256_2_1(f1_p3)  # L_softmax1^P3
        l2_p3 = self.fc_id_256_2_2(f2_p3)  # L_softmax2^P3

        # 3. Multiple Granularity Network 3.1. Network Architecture: During testing phases, to obtain the most powerful
        # discrimination, all the features reduced to 256-dim are concatenated as the final feature.
        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model= mgn = MGN(num_classes=576).to(device)
    # # print(model)
    inputs = torch.randn(8,3,256,256).to(device)
    outputs = model(inputs)
    # # from loss.triplet_loss111 import New_dist
    # # dist = New_dist(outputs[5])
    for i in range(len(outputs)):
        print(i)
        print(outputs[i].shape)