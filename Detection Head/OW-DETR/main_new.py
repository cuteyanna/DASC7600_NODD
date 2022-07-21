import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from backbone import Backbone
from utils import nested_tensor_from_tensor_list
from utils import save_checkpoint, load_checkpoint
from detr import DETR
from dataset import CoCoDataset, detection_collate
from loss_new import DETRLoss
from config import *


coco_train = CoCoDataset(root=img_path, annFile=anno_path, transform=transform)
loader = DataLoader(coco_train, shuffle=True, batch_size=16, collate_fn=detection_collate)

backbone = Backbone('resnet50', train_backbone=False, return_interm_layers=False)
model = DETR(num_cls=91, num_layers=6, embed_size=256, heads=8, dropout=0, forward_expansion=4).to(device)
optim = Adam(model.parameters(), lr=lr)
criterion = DETRLoss(lambda_cls=lambda_cls, lambda_iou=lambda_iou, lambda_L1=lambda_L1)

if load_model:
    model, optim = load_checkpoint(torch.load('last_checkpoint.pth.tar'), model, optim)

tensor_list, targets = next(iter(loader))
for epoch in range(NUM_EPOCH):
    # tensor list means a list of img tensors
    if epoch % 2 == 0 and epoch != 0:
        checkpoint = {'model': model.state_dict(), 'optim': optim.state_dict()}
        save_checkpoint(checkpoint, epoch)

    nested_list = nested_tensor_from_tensor_list(tensor_list)
    # nested_list is samples, [3,2048,8,8]


    h, w = nested_list.mask.shape[1], nested_list.mask.shape[2]

    img_feature = backbone(nested_list)['0'].tensors
    img_feature = img_feature.to(device)
    preds = model(img_feature)
    loss = criterion(img_feature, preds, targets, size=(h, w), topK = 5)
    print(loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()

# test_loader = DataLoader(coco_train, shuffle=True, batch_size=1, collate_fn=detection_collate)
# tensor_list, targets = next(iter(test_loader))
# nested_list = nested_tensor_from_tensor_list(tensor_list)
# test_feature = backbone(nested_list)['0'].tensors
# test_feature = test_feature.to(device)
# test_preds = model(test_feature)
# print('============ preds ============')
# print(test_preds)
# print('============ targets ============')
# print(targets)










