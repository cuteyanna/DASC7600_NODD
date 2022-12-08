import argparse
import os
import sys
root_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(root_path))))
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_utils import str2bool
from project_utils.general_utils import seed_torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from data.get_datasets import get_datasets, get_class_splits
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

from tqdm import tqdm
from config import feature_extract_dir

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_kmeans_semi_sup(merge_test_loader, args, K=None):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes # 20 / 16

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to Old classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (feats, label, idx, mask_lab_) in enumerate(tqdm(merge_test_loader)):

        feats = feats.to(device)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        if args.dataset_name == 'coco':
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.all_classes)-1) # unlabeled 0
                                                     else False for x in label]))
        else:
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                 else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]       # Get unlabelled targets

    print('Fitting Semi-Supervised K-Means...')
    kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                           n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()
    u_targets = u_targets.cpu().numpy()
    np.save(os.path.join(args.save_dir,"all_preds.npy"), all_preds)
    np.save(os.path.join(args.save_dir,"targets.npy"), targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    # Get preds corresponding to unlabelled set
    preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)

    # -----------------------
    # EVALUATE
    # -----------------------
    if args.dataset_name == 'coco':
        return None, None, None, kmeans
    else:
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs,
                                                    save_name='SS-K-Means Train ACC Unlabelled', print_output=True)

        return all_acc, old_acc, new_acc, kmeans


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=180, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--K', default=80, type=int, help='Set manually to run with custom K') #input
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    # parser.add_argument('--warmup_model_exp_id', type=str, default='(30.10.2022__59.822)')
    parser.add_argument('--warmup_model_exp_id', type=str, default='(28.11.2022__00.000)')

    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--spatial', type=str2bool, default=False)
    parser.add_argument('--semi_sup', type=str2bool, default=True)
    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--k_means_init', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='coco', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    # ----------------------
    # INIT
    # ----------------------
    print("begin")
    args = parser.parse_args()
    cluster_accs = {}
    seed_torch(0)
    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')

    args = get_class_splits(args)

    if args.dataset_name == 'coco':
        args.num_labeled_classes = len(args.all_classes) - 1
        args.num_unlabeled_classes = 1
    else:
        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(args.unlabeled_classes) # no used here

    device = torch.device('cuda:0')
    args.device = device
    print(args)

    if args.warmup_model_exp_id is not None:

        args.save_dir += '_' + args.warmup_model_exp_id

        if args.use_best_model:
            args.save_dir += '_best'

        print(f'Using features from experiment: {args.warmup_model_exp_id}')
    else:
        print(f'Using pretrained {args.model_name} features...')

    print(args.save_dir)

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    train_transform, test_transform = None, None
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                             train_transform, test_transform, args)

    # Set target transforms:
    target_transform_dict = {}
    if args.dataset_name != 'coco':
        for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
            target_transform_dict[cls] = i
    else:
        for i, cls in enumerate(list(args.all_classes)):
            target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    # Convert to feature vector dataset
    train_dataset = FeatureVectorDataset(base_dataset=train_dataset, feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset.target_transform = target_transform
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False)


    if args.dataset_name != 'coco':
        test_dataset = FeatureVectorDataset(base_dataset=test_dataset, feature_root=os.path.join(args.save_dir, 'test'))
        unlabelled_train_examples_test = FeatureVectorDataset(base_dataset=unlabelled_train_examples_test,
                                                              feature_root=os.path.join(args.save_dir, 'train'))
        unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                          batch_size=args.batch_size, shuffle=False)

    print('Performing SS-K-Means on all in the training data...')
    all_acc, old_acc, new_acc, kmeans = test_kmeans_semi_sup(train_loader, args, K=args.K)

    print(all_acc, old_acc, new_acc)
    cluster_save_path = os.path.join(args.save_dir, 'ss_kmeans_cluster_centres.pt')
    torch.save(kmeans.cluster_centers_, cluster_save_path)