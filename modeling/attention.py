import torch
import torch.nn as nn
C = 2048

class model(nn.Module):
    def __init__(self, C):
        super(model, self).__init__()
        self.C = C  # 将输入通道数保存为类的属性
        self.conv1 = nn.Conv2d(self.C, self.C // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(self.C // 2, self.C // 2, kernel_size=3, padding=1, groups=32)
        self.bn = nn.BatchNorm2d(self.C // 2)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(self.C // 2, self.C, kernel_size=1)
        self.conv4 = nn.Conv2d(self.C, self.C, kernel_size=3, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(self.C)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bn2(out)
        return out

class model2(nn.Module):
    def __init__(self, C):
        super(model2, self).__init__()
        self.C = C  # 将输入通道数保存为类的属性
        self.conv1 = nn.Conv2d(self.C, self.C // 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.C // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


Module1 = model(C)
Module2= model(C)
Module3 = model2(C)
Module4= model2(C)

class cat(nn.Module):
    def __init__(self,C):
        super(cat, self).__init__()
        # 定义模块1和模块2
        self.module1 = Module1
        self.module2 = Module2
        # 定义关系平均池化层和softmax层
        self.rel_avg_pool = nn.AdaptiveAvgPool2d((C,1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 输入特征依次经过模块1和模块2，并进行相乘
        feat1 = self.module1(x)
        b, c, h, w = feat1.size()
        feat1 = feat1.view(b, c, -1)
        feat2 = self.module2(x)
        b,c,h,w = feat2.size()
        feat2 = feat2.view(b, c,-1 ).transpose(1, 2)
        feat_combine = torch.matmul(feat1 , feat2)
        # 关系平均池化和softmax处理
        feat_pool = self.rel_avg_pool(feat_combine)
        feat_softmax = self.softmax(feat_pool)
        feat_softmax = feat_softmax.unsqueeze(3)
        feat_hadamard = feat_softmax * x
        feat_out = feat_hadamard + x
        return feat_out

class sat(nn.Module):
    def __init__(self,hw):
        super(sat, self).__init__()
        # 定义模块1和模块2
        self.module1 = Module3
        self.module2 = Module4
        # 定义关系平均池化层和softmax层
        self.rel_avg_pool = nn.AdaptiveAvgPool2d((hw,1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 输入特征依次经过模块1和模块2，并进行相乘
        feat1 = self.module1(x)
        feat1 = feat1.view(feat1.size(0),feat1.size(1),-1)
        feat2 = self.module2(x)
        feat2 = feat2.permute(0, 2, 3, 1).contiguous().view(feat2.size(0),-1, feat2.size(1))
        feat_combine = torch.matmul(feat2 , feat1)
        # 关系平均池化和softmax处理
        feat_pool = self.rel_avg_pool(feat_combine)
        feat_softmax = self.softmax(feat_pool)
        # 重新形状重塑
        feat_reshape = feat_softmax.view(x.size(0), -1,x.size(2),x.size(3)).expand(x.size(0), x.size(1),x.size(2),x.size(3))
        # Hadamard乘积和加法
        feat_hadamard = feat_reshape * x
        feat_out = feat_hadamard + x
        return feat_out


import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y+x

import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y





if __name__ == "__main__":
    inputs = torch.randn(8,2048,16,16)
    at1 = cat(2048)
    at2 = sat(16*16)
    # cat = ChannelAttention(2048)
    # # 创建Spatial Attention模块
    # sat = SpatialAttention()
    outputs =at1(inputs)
    outputs =at2(inputs)
    # outputs =cat(inputs)
    # outputs = sat(inputs)
    # b, c, h, w =inputs.size()
    # output1 = inputs.view(b, c,h * w ).transpose(1, 2)
    # output2 = inputs.permute(0, 2, 3, 1).contiguous().view(b,-1, c)
    # print(torch.all(torch.eq(output1, output2)))
