# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
import random
from typing import Iterable

import torch

from timm.utils import accuracy

import utils
from beit.datasets import build_instaformer_dataset
from beit.run_beit_inference import _flatten_list
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def train_class_batch(model, img, boxes, attention_mask, classes, criterion):
    outputs = model(x=img, boxes=boxes, attention_mask=attention_mask)
    loss = criterion(outputs, classes[attention_mask])
    return loss, outputs


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        img, _, attention_mask, boxes_and_labels = batch

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        boxes, classes = boxes_and_labels

        img = img.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        classes = classes.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, output = train_class_batch(model, img=img, boxes=boxes, classes=classes,
                                             attention_mask=attention_mask, criterion=criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        class_acc = (output.max(-1)[-1] == classes[attention_mask]).float().mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_f1_whole(args, model, device):
    dataset = build_instaformer_dataset(args=args, eval_f1=True, data_root=args.eval_data_path)

    predictions = []
    labels = []

    model.eval()
    for i in range(len(dataset)):
        sample, _ = dataset[i]
        img, _, boxes_and_labels = sample
        boxes, classes = boxes_and_labels

        img = img.to(device, non_blocking=True).unsqueeze(0)
        boxes = boxes.float()

        num_boxes = 500
        boxes_split = torch.split(boxes, num_boxes, dim=0)

        with torch.cuda.amp.autocast():
            for b in boxes_split:
                if b.shape[0] == num_boxes:
                    attention_mask = torch.tensor(num_boxes * [True])
                else:
                    padding_length = num_boxes - b.shape[0]
                    attention_mask = torch.tensor(b.shape[0] * [True] + padding_length * [False])
                    fake_box = torch.tensor([-1, -1, -1, -1])
                    fake_box = fake_box.expand(padding_length, -1)
                    b = torch.cat([b, fake_box], dim=0)

                attention_mask = attention_mask.unsqueeze(0).to(device, non_blocking=True)
                b = b.unsqueeze(0).to(device, non_blocking=True)

                logits = model(x=img, boxes=b, attention_mask=attention_mask)
                pred = logits.max(1).indices
                predictions.append(pred)
            labels.append(classes)
    predictions = torch.cat(predictions).cpu()
    labels = torch.cat(labels).cpu()

    types = ['other', 'inflammatory', 'epithelial', 'spindle']
    #types = ['neoplastic ', 'inflammatory', 'soft', 'dead', 'epithelial']

    print(f"All dataset size {labels.shape[0]}")
    print(f"all dataset class F1 {f1_score(labels.numpy(), predictions.numpy(), zero_division=1, average='weighted')}")
    for i in range(len(types)):
        type_samples = (labels == i) | (predictions == i)

        labels_true = labels[type_samples].numpy()
        pred = predictions[type_samples].numpy()

        tp_dt = ((labels_true == i) & (pred == i)).sum()
        tn_dt = ((labels_true != i) & (pred != i)).sum()
        fp_dt = ((labels_true != i) & (pred == i)).sum()
        fn_dt = ((labels_true == i) & (pred != i)).sum()

        f1_type = (2 * tp_dt) / (2 * tp_dt + fp_dt + fn_dt)
        acc_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + fp_dt + fn_dt)
        prec_type = tp_dt / (tp_dt + fp_dt)
        recall_type = tp_dt / (tp_dt + fn_dt)

        print(f"{types[i]} class F1 {f1_type}")
        print(f"{types[i]} class accuracy {acc_type}")
        print(f"{types[i]} class precision {prec_type}")
        print(f"{types[i]} class recall {recall_type}")
        print()
    print(f"Accuracy on the whole ds: {accuracy_score(labels.numpy(), predictions.numpy())}")


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for (batch, _) in metric_logger.log_every(data_loader, 10, header):
        img, _, attention_mask, boxes_and_labels = batch

        boxes, classes = boxes_and_labels

        img = img.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        classes = classes.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(x=img, boxes=boxes, attention_mask=attention_mask)
            loss = criterion(output, classes[attention_mask])

        [acc1] = accuracy(output, classes[attention_mask], topk=(1,))

        batch_size = img.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
