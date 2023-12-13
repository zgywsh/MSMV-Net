# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn,train_collate_fn1, val_collate_fn1
from .datasets import init_dataset, ImageDataset,ImageDataset1
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid,RandomIdentitySampler1  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    # print(cfg.DATASETS.NAMES)
    if len(cfg.DATASETS.NAMES) == 1:
        # print(1)
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        # print(2)
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes


def make_data_loader1(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    # print(cfg.DATASETS.NAMES)
    if len(cfg.DATASETS.NAMES) == 1:
        # print(1)
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        # print(2)
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn1
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler1(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn1
        )

    val_set = ImageDataset1(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn1
    )
    return train_loader, val_loader, len(dataset.query), num_classes

from torch.utils.data import DataLoader

from .datasets import init_dataset, torch_dataset
# from .transforms import build_trasform
from .sampler.triplet import TripletSampler

def build_data(cfg):
    # train_transforms = build_trasform(args, is_train=True)
    # val_transforms = build_trasform(args, is_train=False)
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    dataset = init_dataset(cfg)
    train_set = torch_dataset.ReIDDataset(dataset, train_transforms, is_train=True)
    train_sampler = TripletSampler(dataset.trainset,
                                    cfg.batch_size,
                                    cfg.num_instances)
    train_loader = DataLoader(dataset=train_set,
                                sampler=train_sampler,
                                batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers)
    val_set = torch_dataset.ReIDDataset(dataset, val_transforms, is_train=False)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                            shuffle=False)
    num_train_classes = dataset.num_vids

    return train_loader,val_loader, num_train_classes