# encoding: utf-8
import argparse
import os
import sys
import torch
from torch.nn import functional as F
torch.manual_seed(7)

from layers.triplet_loss import TripletLoss
from layers.center_loss import CenterLoss
import torch.nn as nn

from torch.backends import cudnn
import numpy as np
from PIL import Image

# print(sys.path)
#
sys.path.append('.')
from config import cfg
from data import make_data_loader2
from layers.triplet_loss import euclidean_dist
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from utils.reid_metric import R1_mAP
from engine.trainer import create_supervised_evaluator
import logging
from engine.inference import inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# def test():
#     cudnn.benchmark = True
#
#     from modeling.mgn1 import MGN
#     model = MGN(576).to(device)
#     # 导入模型参数
#     model_params_path = "/home/wangz/verid/checkpoints and logs/mgn1/mgn.pkl"
#     net_params = torch.load(model_params_path)
#     # 把加载的参数应用到模型中
#     model.load_state_dict(net_params,strict=False)
#     AN = []
#     AP = []
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (xb, yb,_) in enumerate(val_loader):
#             # print(yb)
#             yb = torch.tensor(yb)
#             inputs, target = xb.to(device), yb.to(device)
#             outputs = model(inputs)
#             loss_fn = TripletLoss(margin=0.3)
#
#             ap = loss_fn(outputs[1], target)[1]
#             an = loss_fn(outputs[1], target)[2]
#             AN = AN + an[an != 0].tolist()
#             AP = AP + ap[ap != 0].tolist()
#
#     # print(AN)
#     # 打开文件，使用 'w' 模式表示写入
#     file = open('listan.txt', 'w')
#
#     # 将列表逐行写入文件
#     for item in AN :
#         file.write(str(item) + '\n')
#
#     # 关闭文件
#     file.close()
#     file = open('listap.txt', 'w')
#
#     # 将列表逐行写入文件
#     for item in AP :
#         file.write(str(item) + '\n')
#
#     # 关闭文件
#     file.close()
cfg.TEST.IMS_PER_BATCH = 1
train_loader, val_loader, num_query, num_classes = make_data_loader2(cfg)
def getni(numbers, indexes):
    result = [numbers[i] for i in indexes]
    return result

from PIL import Image, ImageOps

def add_border(image, border_size, border_color):
    # 计算带边框的新尺寸
    new_size = (
        image.size[0] + 2 * border_size,
        image.size[1] + 2 * border_size
    )
    # 创建带边框的新图像
    bordered_image = Image.new(image.mode, new_size, border_color)
    # 粘贴原图到带边框的图像中心
    bordered_image.paste(image, (border_size, border_size))
    return bordered_image
def add_border1(image, border_size, border_color):
    # 计算带边框的新尺寸
    new_size = (
        image.size[0] + 2 * border_size,
        # image.size[1] + 2 * border_size
        image.size[1]+ 20
    )
    # 创建带边框的新图像
    bordered_image = Image.new(image.mode, new_size, border_color)
    # 粘贴原图到带边框的图像中心
    bordered_image.paste(image, (border_size, 10))
    return bordered_image

def concatenate_images(images, direction):
    # 计算拼接后图像的宽度和高度
    max_width = max(img.size[0] for img in images)
    max_h = max(img.size[1] for img in images)
    total_height = sum(img.size[1] for img in images)
    total_w = sum(img.size[0]+10 for img in images)

    # 创建拼接后的空白画布
    if direction == 'horizontal':
        concatenated_image = Image.new('RGB', (max_width, total_height),(255,255,255))
    else:  # direction == 'vertical'
        concatenated_image = Image.new('RGB', (total_w, max_h),(255,255,255))

    # 在画布上逐个粘贴图像
    y_offset = 0
    for img in images:
        if direction == 'horizontal':
            concatenated_image.paste(img, (0, y_offset))
            y_offset += img.size[1]
        else:  # direction == 'vertical'
            concatenated_image.paste(img, (y_offset, 0))
            y_offset += (img.size[0]+10)
    return concatenated_image


def test():
    cudnn.benchmark = True

    from modeling.mgn3 import MGN
    model = MGN(576).to(device)
    # 导入模型参数
    model_params_path = "/home/wangz/verid/checkpoints and logs/mgn3/mgn.pkl"
    net_params = torch.load(model_params_path)
    # 把加载的参数应用到模型中
    model.load_state_dict(net_params,strict=False)

    # 将模型设置为评估模式
    model.eval()

    # 禁用梯度计算
    with torch.no_grad():
        border_size = 10
        red = (255, 0, 0)
        green = (0, 255, 0)
        # 红色边框
        direction = 'vertical'  # horizontal或vertical
        new_size = (256, 256)
        # 遍历 val_loader
        for i, (query_images,qy, qcid,qpath,view1) in enumerate(val_loader):
            # if i<5:
            #     continue
            # 将查询图像移动到设备
            query_images = query_images.to(device)
            print(qpath)

            # 获取查询图像的嵌入向量
            query_embeddings = model(query_images)[0]
            dist = [[],[],[],[],[],[],[],[]]
            glabel = [[],[],[],[],[],[],[],[]]
            gpaths = [[],[],[],[],[],[],[],[]]

            images = []

            image = Image.open(qpath[0])
            image = image.resize(new_size)
            image = add_border1(image, 40, (255, 255, 255))
            images.append(image)

            # 遍历图库图像
            for j, (gallery_images,gy, gcid,path,view2) in enumerate(val_loader):
                if gcid ==qcid:
                    continue
                if qpath == path:
                    continue
                # 将图库图像移动到设备
                gallery_images = gallery_images.to(device)

                # 获取图库图像的嵌入向量
                gallery_embeddings = model(gallery_images)[0]

                for v in [(5,), (3,), (4,), (1,), (7,), (0,), (6,), (2,)]:
                    if view2 == v:
                        vi = int(v[0])
                        # 计算查询图像和图库图像之间的相似性分数
                        gpaths[vi].append(path)
                        glabel[vi].append(gy)
                        dist[vi].append(euclidean_dist(query_embeddings, gallery_embeddings))
            pic_path = []
            label = []
            # print(len(dist))
            for vi in range(8):
                # print(len(dist[vi]))
                # sorted_indices = sorted(range(len(dist[vi])), key=lambda k: dist[k])[:1]
                distance = torch.tensor(dist[vi])
                min_index = torch.argmin(distance).item()
                pic_path.append(gpaths[vi][min_index])
                label.append(glabel[vi][min_index])


            # np.save('dist.npy', pic_path)


            for label,path in zip(label,pic_path):

                image = Image.open(path[0])
                image = image.resize(new_size)

                if label == qy:
                    image = add_border(image, border_size, green)
                else:
                    image = add_border(image, border_size, red)

                images.append(image)

            # 拼接图像
            concatenated_image = concatenate_images(images, direction)
            concatenated_image.save("/home/wangz/verid/{}.jpg".format(i))

            if i ==10:
                break


if __name__ == '__main__':
    # main()
    test()




