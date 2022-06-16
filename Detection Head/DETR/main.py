import torch
from backbone import *
from utils import *

sample_1 = torch.randn(3,224,230)
sample_2 = torch.randn(3,230,224)
sample_3 = torch.randn(3,221,213)
tensor_list = [sample_1,sample_2,sample_3]
nested_list = nested_tensor_from_tensor_list(tensor_list)
# nested.mask.shape [3,230,230];three samples
# nested.tensors.shape [3, 3, 230, 230]
backbone = Backbone('resnet50',train_backbone=False,return_interm_layers=False)
out = backbone(nested_list)
print(out.keys())
print(out['0'].tensors.shape)
print(out['0'].mask.shape)

