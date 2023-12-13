# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP
from layers.triplet_loss import TripletLoss
from layers.center_loss import CenterLoss
from config import cfg


global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target,view = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        view = view.to(device) if torch.cuda.device_count() >= 1 else view
        # score, feat = model(img)
        outputs = model(img)
        ####################
        dist = torch.Tensor([[1. , 0.8, 0.2, 0.1, 0., 0.1, 0.2, 0.8],
                             [0.8, 1., 0.7, 0.1, 0.1, 0.1, 0.1, 0.8],
                             [0.2, 0.7, 1., 0.3, 0.1, 0.1, 0.1, 0.2],
                             [0.1, 0.1, 0.3, 1., 0.8, 0.3, 0.1, 0.1],
                             [0. , 0.1, 0.1, 0.8, 1., 0.3, 0.1, 0.1],
                             [0.1, 0.1, 0.1, 0.3, 0.3, 1., 0.7, 0.1],
                             [0.2, 0.1, 0.1, 0.1, 0.1, 0.7, 1., 0.3],
                             [0.8, 0.8, 0.2, 0.1, 0.1, 0.1, 0.3, 1.]]).to(device)

        def view_mat(view):
            view_dist = torch.zeros(len(view), len(view)).to(device)
            # print(view_dist.shape)
            for i in range(len(view)):
                for j in range(i, len(view)):
                    view_dist[i][j] = dist[view[i]][view[j]]
            return view_dist

        _, view_max = torch.max(outputs[-1].data, 1)
        view_dist = view_mat(view_max)
        ##################

        loss_fn = TripletLoss(margin=0.3,if_view = True)
        loss_fn2 = nn.CrossEntropyLoss(label_smoothing = 0.1)

        losses = [loss_fn(output, target,view_dist)[0] for output in outputs[1:4]] + \
                 [loss_fn2(output, target) for output in outputs[4:-1] ]+ \
                 [loss_fn2(outputs[-1], view)]


        loss = sum(losses) / len(losses)

        loss.backward()
        optimizer.step()
        # compute acc
        # acc = (score.max(1)[1] == target).float().mean()
        # return loss.item(), acc.item()
        return loss.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn,
                                          cetner_loss_weight,
                                          device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        # img, target = batch
        # img = img.to(device) if torch.cuda.device_count() >= 1 else img
        # target = target.to(device) if torch.cuda.device_count() >= 1 else target
        # score, feat = model(img)
        # loss = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))

        img, target, view = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        view = view.to(device) if torch.cuda.device_count() >= 1 else view
        # score, feat = model(img)
        outputs = model(img)
        ####################
        dist = torch.Tensor([[1. , 0.8, 0.2, 0.1, 0., 0.1, 0.2, 0.8],
                             [0.8, 1., 0.7, 0.1, 0.1, 0.1, 0.1, 0.8],
                             [0.2, 0.7, 1., 0.3, 0.1, 0.1, 0.1, 0.2],
                             [0.1, 0.1, 0.3, 1., 0.8, 0.3, 0.1, 0.1],
                             [0. , 0.1, 0.1, 0.8, 1., 0.3, 0.1, 0.1],
                             [0.1, 0.1, 0.1, 0.3, 0.3, 1., 0.7, 0.1],
                             [0.2, 0.1, 0.1, 0.1, 0.1, 0.7, 1., 0.3],
                             [0.8, 0.8, 0.2, 0.1, 0.1, 0.1, 0.3, 1.]]).to(device)

        def view_mat(view):
            view_dist = torch.zeros(len(view), len(view)).to(device)
            # print(view_dist.shape)
            for i in range(len(view)):
                for j in range(i, len(view)):
                    view_dist[i][j] = dist[view[i]][view[j]]
            return view_dist

        _, view_max = torch.max(outputs[-1].data, 1)
        view_dist = view_mat(view_max)
        ##################

        # view_dist = outputs[-2]
        loss_fn = TripletLoss(margin=0.3, if_view=True)
        loss_fn2 = nn.CrossEntropyLoss(label_smoothing=0.1)
        # center_criterion = CenterLoss(feat_dim = 256)
        losses = [loss_fn(output, target, view_dist)[0] for output in outputs[1:4]] + \
                 [loss_fn2(output, target) for output in outputs[4:-1]] + \
                 [loss_fn2(outputs[-1], view)] + \
                 [cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(outputs[0], target)]
        loss = sum(losses) / len(losses)

        loss.backward()

        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        # acc = (score.max(1)[1] == target).float().mean()
        # print(loss.item())
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)[0]
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    # checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    # # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
    #                                                                  'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    # RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    # RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch


    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
        #                         scheduler.get_lr()[0]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] , Base Lr: {:.2e}, Loss: {:.3f}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),scheduler.get_lr()[0],engine.state.metrics['avg_loss']))
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] , Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.2%}".format(mAP))

            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
        global Max
        path_state_dict = cfg.OUTPUT_DIR + "/mgn.pkl"
        if engine.state.epoch == 1:
            Max = 0
        if mAP >= Max :
            Max = mAP
            print("max",Max)
            net_state_dict = model.state_dict()
            torch.save(net_state_dict, path_state_dict)
        else:
            net_params = torch.load(path_state_dict)
            # 把加载的参数应用到模型中
            model.load_state_dict(net_params, strict=False)

    trainer.run(train_loader, max_epochs=epochs)



def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    # checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
    #                                                                  'optimizer': optimizer,
    #                                                                  'center_param': center_criterion,
    #                                                                  'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # # average metric to attach on trainer
    # RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    # RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
        #                         scheduler.get_lr()[0]))
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] , Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.2%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))

        global Max
        path_state_dict = cfg.OUTPUT_DIR + "/mgn.pkl"
        if engine.state.epoch == 1:
            Max = 0
        if mAP >= Max:
            Max = mAP
            print("max", Max)
            net_state_dict = model.state_dict()
            torch.save(net_state_dict, path_state_dict)
        else:
            net_params = torch.load(path_state_dict)
            # 把加载的参数应用到模型中
            model.load_state_dict(net_params, strict=False)

    trainer.run(train_loader, max_epochs=epochs)
