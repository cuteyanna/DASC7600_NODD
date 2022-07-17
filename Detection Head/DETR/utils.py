import torch
import os


# combine tensor and mask together
class NestedTensor(object):

    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # casts both tensors and mask to device(CUDA)
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        # returns tensors and mask as a tuple
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
# tensor = torch.randn(2,3,765,911)
# mask = torch.randn(2,765,911)
# nested = NestedTensor(tensor,mask)
# nested.tensors.shape --> torch.Size([2,3,765,911])
# nested.mask.shape --> torch.Size([2,765,911])


def _max_by_axis(the_list):
    # Input: [[3, 224, 230], [3, 230, 224]]
    # Output: [3,230,230]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # img 是tensor_list里面的原始图片;torch.Size([3, 224, 230])
            # pad_img 是填补零以后的图片/tensor，初始全为0；torch.Size([3, 230, 230])
            # m是mask的tensor，没有Chanel维度，初始全为1；torch.Size([230, 230]) mask值=0 设置为True
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def save_checkpoint(checkpoint, epoch, root_dir='checkpoints/', filename='last_checkpoint.pth.tar', keep_all=False):
    print('===> Saving the checkpoint')

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if not keep_all:
        torch.save(checkpoint, root_dir + filename)
    else:
        filename = 'epoch_{}_checkpoint.pth.tar'.format(epoch)
        torch.save(checkpoint, root_dir + filename)


def load_checkpoint(checkpoint, model, optim):
    print('===> Loading the checkpoint')

    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])

    return model, optim

