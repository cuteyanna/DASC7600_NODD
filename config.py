# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '/content/drive/My Drive/capstone/GCD/datasets/cifar10'
cifar_100_root = '/content/drive/My Drive/capstone/GCD/datasets/cifar100'
cub_root = '/content/drive/My Drive/capstone/GCD/datasets/CUB'
aircraft_root = '/content/drive/My Drive/capstone/GCD/datasets/aircraft/fgvc-aircraft-2013b'
herbarium_dataroot = '/content/drive/My Drive/capstone/GCD/datasets/herbarium_19/'
imagenet_root = '/content/drive/My Drive/capstone/GCD/ImageNet/ILSVRC12'

# OSR Split dir
osr_split_dir = '\data\ssb_splits'

# -----------------
# OTHER PATHS
# -----------------

dino_pretrain_path = './dino_vitbase16_pretrain.pth'
dino_pretrain_path_coco = './dino_vitbase16_pretrain.pth'#'./osr_novel_categories/metric_learn_gcd/log/(04.11.2022__41.393)/checkpoints/model.pt'

feature_extract_dir = './osr_novel_categories/extracted_features_public_impl'     # Extract features to this directory
exp_root = './osr_novel_categories/'          # All logs and checkpoints will be saved here