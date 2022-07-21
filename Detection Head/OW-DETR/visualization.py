# Import the required libraries
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from box_ops import box_cxcywh_to_xyxy


def visualization(preds, img_name, root_dir='Data/val2017/'):
    # read input image from your computer
    img_path = root_dir + img_name
    img = read_image(img_path)
    # pred_logits = preds['pred_logits']
    pred_boxes = preds['pred_boxes']
    box = box_cxcywh_to_xyxy(pred_boxes)
    img = draw_bounding_boxes(img, box, width=5, colors="green", fill=False)
    # transform this image to PIL image
    img = torchvision.transforms.ToPILImage()(img)

    # display output
    img.show()


if __name__ == '__main__':
    img_name = '000000000139.jpg'
    preds = {'pred_boxes': torch.tensor([[200, 200, 300, 300], [385, 412, 455, 510]])}
    visualization(preds, img_name)



