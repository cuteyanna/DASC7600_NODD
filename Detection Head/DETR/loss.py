import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from utils import *


class DETRLoss(nn.Module):
    def __init__(self, lambda_cls, lambda_iou, lambda_L1):
        super(DETRLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_iou = lambda_iou
        self.lambda_L1 = lambda_L1

    def forward(self, outputs, targets):
        # "pred_logits": (N, num_queries, num_class+1)
        # "pred_boxes" : (N, num_queries, 4)

        # "labels": [(num_targets) for _ in range(N)] -> A list contains N tensor,
        #                                                each tensor represents a single picture's targets
        # "boxes" : [(num_targets, 4) for _ in range(N)] -> Similar to the labels

        # {'label':[tensor, tensor, tensor, ], 'boxes':[]}
        # len(tgt['label]) = N
        # tensor : (num_targets)
        N, num_queries = outputs["pred_logits"].shape[0], outputs["pred_logits"].shape[1]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Then we concat all the targets together for the convenience of computation
        # We would split them later, no worries about the batch-crossed linkage (Wont be used later)
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # calculate probability here, measure how accurate the class is
        # we want to minimize (1 - out_prob(of the gt class)) (out_prob->1)->(loss->0)
        cost_class = -out_prob[:, tgt_ids]  # (N*num_queries, len(tgt_ids))
        # Compute the L1 cost between boxes
        # p indicates using L1Norm, Shape: (N*num_queries, len(tgt_ids))
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        Cost = self.lambda_L1 * cost_bbox + self.lambda_cls * cost_class + self.lambda_iou * cost_giou
        Cost = Cost.reshape(N, num_queries, -1)
        C = Cost.cpu()

        sizes = [len(v["boxes"]) for v in targets]
        multiplier = torch.zeros_like(Cost)
        for i, c in enumerate(C.split(sizes, -1)):
            count = 0  # since we use C.split, the actual idx = count + idx
            # we got the row_idx and col_idx of the matched bbox
            row_idx, col_idx = linear_sum_assignment(c[i])
            multiplier[i, row_idx, col_idx + count] = 1
            # Use a multiplier matrix with 0 on not matched position and 1 for matched position

            # Update count number
            count += sizes[i]
        loss = torch.sum(Cost * multiplier)

        return loss


if __name__ == '__main__':
    import numpy as np
    N = 16
    num_queries = 100
    num_class = 70

    preds = {'pred_logits': torch.randn(N, num_queries, num_class+1),
             'pred_boxes': torch.randn(N, num_queries, 4).sigmoid()}

    targets = []
    for i in range(N):
        tmp = {}
        num_obj = np.random.randint(2, 20)
        tmp['labels'] = torch.randint(0, 70, [num_obj])
        boxes = torch.randn(num_obj, 4)
        # Add here just to make sure the right bottom corner value is larger than the left top
        boxes[..., 2:] += 1
        tmp['boxes'] = boxes.sigmoid()
        targets.append(tmp)

    loss_fn = DETRLoss(1, 1, 1)
    print(loss_fn(preds, targets))









