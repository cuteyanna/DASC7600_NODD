import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import draw_bounding_boxes

from backbone import Backbone
from box_ops import rescale_bboxes
from config import *
from dataset import CoCoDataset, detection_collate
from detr import DETR
from utils import load_checkpoint
from utils import nested_tensor_from_tensor_list

import numpy as np

model = DETR(num_cls=91, num_layers=6, embed_size=256, heads=8, dropout=0, forward_expansion=4).to(device)
optim = Adam(model.parameters(), lr=lr)

model, optim = load_checkpoint(torch.load('checkpoints/last_checkpoint.pth.tar'), model, optim)
backbone = Backbone('resnet50', train_backbone=False, return_interm_layers=False)

coco_train = CoCoDataset(root=img_path, annFile=anno_path, transform=transform)
loader = DataLoader(coco_train, shuffle=False, batch_size=16, collate_fn=detection_collate)

tensor_list, targets = next(iter(loader))

model.eval()
nested_list = nested_tensor_from_tensor_list(tensor_list)
h, w = nested_list.mask.shape[1], nested_list.mask.shape[2]

img_feature = backbone(nested_list)['0'].tensors  # torch.Size([16, 2048, 20, 20])
img_feature = img_feature.to(device)
preds = model(img_feature)

for i in range(16):
    count = 0
    image = torch.tensor(tensor_list[i]*255, dtype=torch.uint8)

    for j in range(100):
        if preds['pred_logits'][i][j][-1].sigmoid() >= 0.983:
            box = rescale_bboxes(preds['pred_boxes'][i][j], size=(w, h))
            box = box.unsqueeze(0)
            image = draw_bounding_boxes(image, box, width=5, colors="green", fill=False)

    img = torchvision.transforms.ToPILImage()(image)
    img.show()
