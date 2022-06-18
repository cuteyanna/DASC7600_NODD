import torch
from torchvision import transforms

import albumentations as A


device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCH = 20
lr = 1e-4
img_path = 'Data/val2017'
anno_path = 'Data/annotations_trainval2017/annotations/instances_val2017.json'

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Loss parameter
lambda_cls = 1
lambda_iou = 1
lambda_L1  = 1
