import torch


def box_cxcywh_to_xyxy(x):
    """
    :param x: bounding box with (x_center, y_center, width, height) format
    :return: bounding box with (x1, y1, x2, y2) format
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox,size):
    # size is img.size
    img_w, img_h = size
    # convert cxcy to xyxy
    xyxy = box_cxcywh_to_xyxy(out_bbox)
    b = xyxy * torch.tensor([img_w,img_h,img_w,img_h], dtype = torch.float32)
    return b


def box_iou(box1, box2):
    # get the value of box1
    box1_x1 = box1[..., 0:1]  # (N, 4) -> (N, 1)
    box1_y1 = box1[..., 1:2]
    box1_x2 = box1[..., 2:3]
    box1_y2 = box1[..., 3:4]

    # get the value of box2
    box2_x1 = box2[..., 0:1]
    box2_y1 = box2[..., 1:2]
    box2_x2 = box2[..., 2:3]
    box2_y2 = box2[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) for the case that no intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersection / (box1_area + box2_area - intersection + 1e-6)
    return iou


def generalized_box_iou(box1, box2):
    """
    :param box1: out bbox with shape of (N*num_queries, 4)
    :param box2: target bbox with shape of (S, 4), S represent the sum of targets in this Batch
    :return: bbox iou for each pair of box1 and box2 with shape of (N*num_queries, S)
    """
    # Use broadcast to calculate the pairwise iou
    iou_mat = box_iou(box1.unsqueeze(1), box2.unsqueeze(0))
    return iou_mat.squeeze(-1)


if __name__ == '__main__':
    test_box1 = torch.as_tensor([[0.1, 0.1, 0.3, 0.3], [0.4, 0.3, 0.8, 0.8]])
    test_box2 = torch.as_tensor([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.8, 0.8], [0.2, 0.2, 0.9, 0.9]])
    print(generalized_box_iou(test_box1, test_box2))
