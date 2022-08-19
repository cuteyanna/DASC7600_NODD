from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import numpy as np

import albumentations as A


class CoCoDataset(CocoDetection):
    def __init__(self, root, annFile, transform):
        super(CoCoDataset, self).__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img, tgt = super(CoCoDataset, self).__getitem__(idx)
        tgt_dict = {'labels': torch.empty(0, dtype=torch.long), 'boxes': torch.empty(0)}
        # {'label':tensor, 'boxes':tensor}
        for item in tgt:
            item['category_id'] = torch.as_tensor([item['category_id']])
            item['bbox'] = torch.as_tensor(item['bbox']).unsqueeze(0)

            tgt_dict['labels'] = torch.cat([tgt_dict['labels'], item['category_id']])
            tgt_dict['boxes'] = torch.cat([tgt_dict['boxes'], item['bbox']])

        if self.transform is not None:
            img = self.transform(img)

        return img_id, (img, tgt_dict)


def detection_collate(batch):
    """
    :param batch: A batch contain imgs and annotation
    :return: (list of image, list of annotation)
    """
    img_ids = []
    imgs = []
    targets = []

    for img_id, (img, tgt) in batch:
        img_ids.append(img_id)
        imgs.append(img)
        targets.append(tgt)
    return img_ids, imgs, targets


if __name__ == '__main__':
    # dir of images and annotation
    img_path = 'Data/val2017'
    anno_path = 'Data/annotations_trainval2017/annotations/instances_val2017.json'

    # define transform for images
    # Use
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    coco_train = CoCoDataset(root=img_path, annFile=anno_path, transform=transform)
    loader = DataLoader(coco_train, shuffle=False, batch_size=4, collate_fn=detection_collate)
    img_ids, imgs, targets = next(iter(loader))
    print(img_ids)
    print(imgs)
    print(targets)
    # [{'boxes': torch.size([6, 4])}, {}, {},...]
