import torch
from sklearn.cluster import KMeans
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import argparse
import os
import sys
import torch
from config import cfg

from torch.backends import cudnn
# print(sys.path)
#
sys.path.append('.')
# par_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(par_dir)
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR

from utils.logger import setup_logger
from modeling.mgn_resnet50 import MGN

def kmeans(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = MGN(576).to(device)
    # 导入模型参数
    model_params_path = "/home/wangz/BNN/checkpoints and logs/mgn_bn/mgn_bn.pkl"
    net_params = torch.load(model_params_path)
    # 把加载的参数应用到模型中
    model.load_state_dict(net_params, strict=False)
    model.eval()
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(train_loader):
            print(batch_idx)
            inputs, labels = xb.to(device), yb.to(device)
            outputs = model(inputs)[0]
            if batch_idx==0:
                data=outputs
            t_new=outputs
            data=torch.cat([data, t_new], dim=0)
    return data

if __name__ == "__main__":
    data=kmeans(cfg)
    print(data.shape)
    torch.save(data, 'data.pth')
    # data = torch.randn(36160, 2048).to(device)

    # 调用K-Means算法进行聚类
    kmeans = KMeans(n_clusters=5, init="k-means++",random_state=0,n_init=10)
    # 样本所属的簇标签
    cluster_labels = kmeans.fit_predict(data.detach().cpu().numpy())
    torch.save(cluster_labels, 'cluster_labels.pth')
    #簇中心tensor
    center_f=kmeans.cluster_centers_
    torch.save(center_f, 'center_f.pth')

    # v_tensor = torch.load('center_f.pth')
    # print(v_tensor[0])