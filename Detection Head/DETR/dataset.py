from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms

import torch


img_path = 'Data/val2017'
anno_path = 'Data/annotations_trainval2017/annotations/instances_val2017.json'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
coco_train = CocoDetection(root=img_path, annFile=anno_path, transform=transform)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for img, tgt in batch:
        imgs.append(img)
        targets.append(tgt)
    return imgs, targets


loader = DataLoader(coco_train, shuffle=True, batch_size=16, collate_fn=detection_collate)
imgs, targets = next(iter(loader))
print(imgs, targets)
print(imgs)
