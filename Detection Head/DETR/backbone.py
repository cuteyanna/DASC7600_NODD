import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from utils import NestedTensor


class BackboneBase(nn.Module):

    def __init__(self,backbone,train_backbone,num_channels,return_interm_layers):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                # if not train the backbone, the parameters dose nor require gradient
                # if train the backbone, back propagate to the second layer(for resnet-50)
                parameter.requires_grad_(False)
        # whether return intermediate layers or not
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self,tensor_list):
        xs = self.body(tensor_list.tensors)
        out = {}
        m = tensor_list.mask
        assert m is not None
        for name, x in xs.items():
            # size是x的最后两个维度，也就是W*H，m[None]就是unsqueeze(0),增加一个维度在dim=0；m[:,None]增加一个维度在dim=1
            # 对m进行下采样/上采样，input：torch.Size([1, 3, 230, 230])；output：torch.Size([1, 3, 8, 8])，且输出都为true和false
            # 因为不同的layers的维度不同，此时mask也要相对因的进行interpolate
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(self, name,train_backbone,return_interm_layers,):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, False],
            pretrained=False, )
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
