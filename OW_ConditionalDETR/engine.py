# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from datetime import datetime
from copy import deepcopy
from collections import Counter

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from torchvision.ops import batched_nms
import numpy as np

import util.misc as utils
from util import box_ops
from datasets.closed_world_eval import Voc_Evaluator
from datasets.voc import Voc_GT


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, nc_epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'image_id'} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(samples, outputs, targets, epoch)
        weight_dict = deepcopy(criterion.weight_dict)

        ######################################
        if epoch < nc_epoch:
            for k, v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0
        ######################################

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        ######################################
        step = epoch * len(data_loader) + idx
        for loss_key, loss_value in loss_dict.items():
            writer.add_scalar(loss_key, loss_value, step)
        ######################################
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, base_ds, postprocessors, data_loader, device):
    model.eval()
    # criterion.eval()

    print('Start Testing')
    voc_evaluator = Voc_Evaluator(base_ds)

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = {k: v.to(device) if k != 'image_id' else v for k, v in targets[0].items()}

        outputs = model(samples)

        orig_target_sizes = torch.stack([targets["orig_size"]], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {targets['image_id']: results}

        voc_evaluator.update(res)

    voc_evaluator.accumulate()
    voc_evaluator.summarize()
    '''
    res
    {'2007_000039.jpg':[{'scores':torch.Size([100]),'labels':torch.Size([100]),'boxes':torch.Size([100, 4])}]}
    '''


@torch.no_grad()
def visualization(model, data_loader, device, conf_thres=0.5, iou_thres=0.45, sample_ratio=0.1):
    print('Starting Visualization')
    model.eval()
    output_dir = 'Results/visuals/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ####### draw feature pics #######
    feature_dir = 'Results/feature_res/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M"))

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    ####### draw feature pics #######

    sample_size = int(len(data_loader) * sample_ratio)
    sample_indices = np.random.choice(np.arange(len(data_loader)), size=sample_size)

    count = []
    for idx, (samples, targets) in enumerate(data_loader):
        if idx in sample_indices:
            samples = samples.to(device)
            targets = {k: v.to(device) if k != 'image_id' else v for k, v in targets[0].items()}

            outputs = model(samples)

            probas = outputs['pred_logits'].softmax(-1)[0, :, :]
            obj_score = outputs['pred_nc_logits'].reshape(-1).sigmoid()
            keep = probas.max(-1).values * obj_score > conf_thres
            # keep = probas.max(-1).values > conf_thres

            scores = outputs['pred_logits'][0, keep].max(-1).values.sigmoid()
            labels = outputs['pred_logits'][0, keep].argmax(-1)
            # labels = labels.cpu().numpy().tolist()

            orig_target_sizes = targets["orig_size"]
            predictied_boxes = outputs['pred_boxes'][0, keep]
            boxes = box_ops.box_cxcywh_to_xyxy(predictied_boxes)
            img_h, img_w = orig_target_sizes.unbind()
            boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h]).to(device)  # convert to original size

            ##################################
            i = batched_nms(boxes, scores, labels, iou_thres)  # batched nms operation
            boxes = boxes[i]
            labels = labels[i]
            # labels = labels.cpu().numpy().tolist()
            count.extend(labels.cpu().numpy().tolist())
            ##################################

            box_ops.plot_bbox(root_path='voc_data/images/test/',
                              image_path=targets.get('image_id'),
                              boxes=boxes,
                              labels=labels,
                              img_output_dir=output_dir)

            ####### draw feature pics #######
            feat = torch.mean(outputs['resnet_1024_feat'], 1)
            upsample = nn.Upsample(size=(img_h, img_w), mode='bilinear')
            img_feat = upsample(feat.unsqueeze(0)).squeeze(0)
            box_ops.plot_feat(img_feat,
                              boxes=boxes,
                              labels=labels,
                              image_path=targets.get('image_id'),
                              img_output_dir=feature_dir)
            ####### draw feature pics #######

        else:
            continue

    # print detected obj info
    counter = Counter(count)
    for k in sorted(counter.keys()):
        print('class {} has {} objects detected'.format(k, counter[k]))


@torch.no_grad()
def detect(model, root_path, device, save_img=True, conf_thres=0.5, iou_thres=0.45):
    print('Starting Inference')
    model.eval()

    def write_prediction(labels, boxes, scores, lbs_path):
        labels = labels.cpu().numpy().tolist()

        with open(lbs_path, mode='w') as f:
            for label, box, score in zip(labels, boxes, scores):
                f.write('{} {} {}\n'.format(str(label),
                                            ' '.join(str(b) for b in box.cpu().numpy().tolist()),
                                            str(score.item())))

    img_list = os.listdir(root_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    idx_to_labels = Voc_GT.CLASS_NAMES

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    img_output_dir = 'Results/detect/{}/images/'.format(time_stamp)
    lbs_output_dir = 'Results/detect/{}/labels/'.format(time_stamp)

    count = []
    for idx, image_path in enumerate(img_list):

        if idx % 1000 == 0 and idx != 0:
            print('{} / {} pictures has been detected ...'.format(idx, len(img_list)))

        img = Image.open(os.path.join(root_path, image_path)).convert('RGB')
        img_h, img_w = img.height, img.width
        img = transform(img)
        img = [img.to(device)]  # for nest tensor operation

        outputs = model(img)

        probas = outputs['pred_logits'].softmax(-1)[0, :, :]
        obj_score = outputs['pred_nc_logits'].reshape(-1).sigmoid()
        keep = probas.max(-1).values * obj_score > conf_thres
        # keep = probas.max(-1).values > conf_thres

        scores = outputs['pred_logits'][0, keep].max(-1).values.sigmoid()
        labels = outputs['pred_logits'][0, keep].argmax(-1)
        # labels = labels.cpu().numpy().tolist()

        predictied_boxes = outputs['pred_boxes'][0, keep]
        boxes = box_ops.box_cxcywh_to_xyxy(predictied_boxes)
        boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h]).to(device)

        ##################################
        i = batched_nms(boxes, scores, labels, iou_thres)  # batched nms operation
        boxes = boxes[i]
        labels = labels[i]
        scores = scores[i]
        # labels = labels.cpu().numpy().tolist()
        count.extend(labels.cpu().numpy().tolist())
        ##################################

        if save_img:
            if not os.path.exists(img_output_dir):
                os.makedirs(img_output_dir)

            box_ops.plot_bbox(root_path, image_path, boxes, labels, img_output_dir)

        img_name, _ = os.path.splitext(image_path)
        if not os.path.exists(lbs_output_dir):
            os.makedirs(lbs_output_dir)

        write_prediction(labels, boxes, scores, os.path.join(lbs_output_dir, img_name + '.txt'))

    # print detected obj info
    counter = Counter(count)
    for k in sorted(counter.keys()):
        print('class {} has {} objects detected'.format(k, counter[k]))
