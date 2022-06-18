import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from backbone import Backbone
from utils import nested_tensor_from_tensor_list
from detr import DETR
from dataset import CoCoDataset, detection_collate
from loss import DETRLoss
from config import *


coco_train = CoCoDataset(root=img_path, annFile=anno_path, transform=transform)
loader = DataLoader(coco_train, shuffle=True, batch_size=16, collate_fn=detection_collate)

backbone = Backbone('resnet50', train_backbone=False, return_interm_layers=False)
model = DETR(num_cls=91, num_layers=6, embed_size=256, heads=8, dropout=0, forward_expansion=4)
optim = Adam(model.parameters(), lr=lr)
criterion = DETRLoss(lambda_cls=lambda_cls, lambda_iou=lambda_iou, lambda_L1=lambda_L1)

tensor_list, targets = next(iter(loader))
for epoch in range(NUM_EPOCH):
    # tensor list means a list of img tensors

    nested_list, (h, w) = nested_tensor_from_tensor_list(tensor_list)
    img_feature = backbone(nested_list)['0'].tensors
    preds = model(img_feature)
    loss = criterion(preds, targets, size=(w, h))
    print(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()

    # print(loss.item())







