import argparse
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from datasets import transforms


class VocDateset(Dataset):
    """
    The annotations and images are structured as follows
    VOC/
        annotation/
            train/
            test/
        images/
            train/
            test/
    """

    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.anno_root_dir = os.path.join(root_dir, 'annotation')
        self.anno_paths = self.get_paths(self.anno_root_dir, self.image_set)

        self.img_root_dir = os.path.join(self.root_dir, 'images')
        self.img_paths = self.get_paths(self.img_root_dir, self.image_set)

        self.transform = transform

    @staticmethod
    def get_paths(root_dir, mode):
        mode_root_dir = os.path.join(root_dir, mode)
        return [os.path.join(mode_root_dir, anno) for anno in os.listdir(mode_root_dir)]

    @staticmethod
    def parse_json(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def prepare_tgt(tgt):
        tgt_dict = {'labels': torch.empty(0, dtype=torch.long), 'boxes': torch.empty(0)}
        tgt_dict.update({'image_id': tgt['images'].get('file_name')})
        tgt_dict.update({'orig_size': torch.as_tensor([tgt['images'].get('height'), tgt['images'].get('width')])})
        # {'label':tensor, 'boxes':tensor}
        for item in tgt.get('annotations'):
            item['category_id'] = torch.as_tensor([item['category_id']])
            item['bbox'] = torch.as_tensor(item['bbox']).reshape(-1, 4)
            # item['difficult'] = torch.as_tensor(item['difficult'])

            tgt_dict['labels'] = torch.cat([tgt_dict['labels'], item['category_id']])
            tgt_dict['boxes'] = torch.cat([tgt_dict['boxes'], item['bbox']])
            # tgt_dict['difficult'] = torch.cat([tgt_dict['difficult'], item['difficult']])

        return tgt_dict

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        tgt = self.parse_json(self.anno_paths[idx])
        tgt = self.prepare_tgt(tgt)

        if self.transform is not None:
            img, tgt = self.transform(img, tgt)

        return img, tgt

    def __len__(self):
        return len(self.anno_paths)


def make_voc_transforms(image_set='train'):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomSelect(
                transforms.RandomResize(scales, max_size=1333),
                transforms.Compose([
                    transforms.RandomResize([400, 500, 600]),
                    # transforms.RandomSizeCrop(384, 600),
                    transforms.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    elif image_set == 'test':
        return transforms.Compose([
            transforms.RandomResize([800], max_size=1333),
            normalize,
        ])

    else:
        raise ValueError('mode could only be train or test')


def build(image_set, args):
    root_dir = args.voc_path
    dataset = VocDateset(root_dir=root_dir, image_set=image_set, transform=make_voc_transforms(image_set))
    return dataset


def draw_bbox(tgt):
    import torchvision
    img = read_image(os.path.join('../voc_data/images/train/', tgt.get('image_id')))
    img = draw_bounding_boxes(img, tgt.get('boxes'))
    img = torchvision.transforms.ToPILImage()(img)
    img.show()


class Voc_GT(object):
    # CLASS_NAMES = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7,
    #                'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14,
    #                'tvmonitor': 15, 'pottedplant': 16, 'unknown': 17}

    CLASS_NAMES = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus',
                   6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog',
                   12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant',
                   16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor', 20: 'unknown'}

    image_set = 'voc_data/images/test'
    annotations = 'voc_data/annotation/test'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    args = parser.parse_args()
    args.voc_path = '../voc_data/'
    train_set = build(image_set='train', args=args)
    # loader = DataLoader(train_set, )
    # count = 0
    # for idx, (_, tgt) in enumerate(train_set):
    #     print(idx)
    #     if len(tgt['boxes']) == 0:
    #         count += 1
    # print(count)

    # voc_gt = Voc_GT()
    # print(voc_gt.CLASS_NAMES)
    # print(voc_gt.image_set)
    # print(voc_gt.annotations)
