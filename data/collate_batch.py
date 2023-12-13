# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


# def train_collate_fn(batch):
#     imgs, pids, camids, img_path = zip(*batch)
#     pids = torch.tensor(pids, dtype=torch.int64)
#     return torch.stack(imgs, dim=0), pids, camids, img_path
#
#
# def val_collate_fn(batch):
#     imgs, pids, camids, img_path = zip(*batch)
#     return torch.stack(imgs, dim=0), pids, camids, img_path

def train_collate_fn(batch):
    imgs, pids, camid,view = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    view = torch.tensor(view, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, view
def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def train_collate_fn1(batch):
    imgs, pids, camid = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids
def val_collate_fn1(batch):
    imgs, pids, camids= zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids