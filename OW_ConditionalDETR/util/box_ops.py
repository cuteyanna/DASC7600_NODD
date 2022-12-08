# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import os

import numpy as np
import torch
from torchvision.ops.boxes import box_area
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from datasets.voc import Voc_GT


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def plot_bbox(root_path, image_path, boxes, labels, img_output_dir):
    labels = labels.cpu().numpy()

    idx_to_labels = Voc_GT.CLASS_NAMES
    labels_to_colors = {0: '#7dfc00', 1: '#228c68', 2: '#8ad8e8', 3: '#235b54', 4: '#29bdab',
                        5: '#3998f5', 6: '#37294f', 7: '#277da7', 8: '#3750db', 9: '#f22020',
                        10: '#ffcba5', 11: '#e68f66', 12: '#632819', 13: '#ffc413', 14: '#f47a22',
                        15: '#2f2aa0', 16: '#b732cc', 17: '#772b9d', 18: '#f07cab', 19: '#d30b94',
                        20: '#edeff3'}

    img = read_image(os.path.join(root_path, image_path))
    for lbs in np.unique(labels):
        img = draw_bounding_boxes(img,
                                  boxes[labels == lbs],
                                  [idx_to_labels[lbs]] * (labels == lbs).sum(),
                                  colors=labels_to_colors[lbs],
                                  width=4,
                                  font_size=13)

    img = transforms.ToPILImage()(img)
    img.save(os.path.join(img_output_dir, image_path))


def plot_feat(img, boxes, labels, image_path, img_output_dir):
    img *= 255
    img = img.cpu().type(torch.uint8)
    labels = labels.cpu().numpy()

    idx_to_labels = Voc_GT.CLASS_NAMES
    labels_to_colors = {0: '#7dfc00', 1: '#228c68', 2: '#8ad8e8', 3: '#235b54', 4: '#29bdab',
                        5: '#3998f5', 6: '#37294f', 7: '#277da7', 8: '#3750db', 9: '#f22020',
                        10: '#ffcba5', 11: '#e68f66', 12: '#632819', 13: '#ffc413', 14: '#f47a22',
                        15: '#2f2aa0', 16: '#b732cc', 17: '#772b9d', 18: '#f07cab', 19: '#d30b94',
                        20: '#edeff3'}
    for lbs in np.unique(labels):
        img = draw_bounding_boxes(img,
                                  boxes[labels == lbs],
                                  [idx_to_labels[lbs]] * (labels == lbs).sum(),
                                  colors=labels_to_colors[lbs],
                                  width=4,
                                  font_size=13)
    img = transforms.ToPILImage()(img)

    img.save(os.path.join(img_output_dir, image_path))

