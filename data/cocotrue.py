import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import platform
from data.data_utils import subsample_instances
plat = platform.system().lower()
if plat == 'linux':
    car_root = "./datasets/coco/{}/"
else:
    car_root = "./datasets/coco/coco_train_val/{}/"


meta_default_path = "./datasets/coco/devkit/coco_{}.mat"

class CocoTrueDataset(Dataset):
    """
        Coco Dataset
    """
    def __init__(self, train=True, limit=None, data_dir=car_root, transform=None, unlabel_reserved = None
                 ,metas=meta_default_path):
        """wwa"""
        metas = metas.format('train_60_class_balanced') if train else metas.format('test_60_class_balanced')
        # metas = metas.format('train_60_class') if train else metas.format('test_60_class')

        # metas = metas.format('train_60_class_high_quality') if train else metas.format('test_60_class_high_quality')
        # metas = metas.format('train_annos_80') if train else metas.format('test_annos_80')
        data_dir = data_dir.format('val2017') if train else data_dir.format('val2017')
        # data_dir = data_dir.format('val2017') if train else data_dir.format('test')

        self.loader = default_loader
        self.unlabel_reserved =list(range(1,21+unlabel_reserved)) if unlabel_reserved else None
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.box = []
        self.train = train
        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)
        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            b = [x[0][0] for x in img_[:4]]
            # if b[3]-b[1] < 2 or b[2]-b[0] < 2 or (b[3]-b[1])*(b[2]-b[0])<500:
            if b[3] - b[1] < 2 or b[2] - b[0] < 2:
                print(img_,"null")
                continue
            if self.unlabel_reserved and int(img_[4][0]) not in self.unlabel_reserved:
                continue
            self.data.append(data_dir + img_[5][0])
            self.target.append(int(img_[4][0]))
            self.box.append(b)

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1 #0~79
        box = self.box[idx]
        image = image.crop(box)
        # image.show()

        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                print(self.data[idx],target,box)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.box = np.array(dataset.box)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def subsample_classes(dataset, include_classes=range(20)):

    # include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    include_classes_cars = [1,2,3,4,5] # labelled classes
    include_classes_coco = np.array(include_classes) + 1# 1~20, unlabelled:21
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_coco]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cocotrue_datasets(train_transform, test_transform, train_classes=range(20), prop_train_labels=0.8,
                    unlabel_reserved = None, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CocoTrueDataset(data_dir=car_root, limit=None, unlabel_reserved=unlabel_reserved, transform=train_transform, metas=meta_default_path, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes) #126680
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices) #train label 抽样

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CocoTrueDataset(data_dir=car_root, limit=None, transform=test_transform, unlabel_reserved = unlabel_reserved, metas=meta_default_path, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled, #63340
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

if __name__ == '__main__':
    import sys
    root_path = os.path.abspath(__file__)
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(root_path))))

    x = get_cocotrue_datasets(None, None, train_classes=range(20), unlabel_reserved = 20, prop_train_labels=0.5, split_train_val=False)
    # x = get_scars_datasets(None, None, train_classes=range(98), prop_train_labels=0.5, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')