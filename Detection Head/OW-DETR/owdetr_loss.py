import torch
import torch.nn as nn

import numpy as np
from scipy.optimize import linear_sum_assignment

from box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou, bbox_normalize


class HungarianMatcher(nn.Module):
    def __init__(self, lambda_cls, lambda_iou, lambda_L1):
        super(HungarianMatcher, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_iou = lambda_iou
        self.lambda_L1 = lambda_L1

    @torch.no_grad()
    def forward(self, outputs, targets, size):
        # "pred_logits": (N, num_queries, num_class+1)
        # "pred_boxes" : (N, num_queries, 4)

        # "labels": [(num_targets) for _ in range(N)] -> A list contains N tensor,
        #                                                each tensor represents a single picture's targets
        # "boxes" : [(num_targets, 4) for _ in range(N)] -> Similar to the labels

        # {'label':[tensor, tensor, tensor, ], 'boxes':[]}
        # len(tgt['label]) = N
        # tensor : (num_targets)
        N, num_queries = outputs["pred_logits"].shape[0], outputs["pred_logits"].shape[1]
        device = outputs["pred_logits"].device

        # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_class+1]
        out_prob = torch.cat([outputs["pred_logits"].flatten(0, 1)[..., :-1].softmax(-1),
                              outputs["pred_logits"].flatten(0, 1)[..., -1:].sigmoid()], dim=-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Then we concat all the targets together for the convenience of computation
        # We would split them later, no worries about the batch-crossed linkage (Wont be used later)
        # Also concat the target labels and boxes
        # [{'labels': tensor(num_targets), 'boxes': tensor(num_targets, 4)}, {...}, {...}, {...}]
        tgt_ids = torch.cat([v["labels"] for v in targets]).to(device)
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).to(device)

        # calculate probability here, measure how accurate the class is
        # we want to minimize (1 - out_prob(of the gt class)) (out_prob->1)->(loss->0)
        cost_class = -out_prob[:, tgt_ids]  # (N*num_queries, len(tgt_ids))
        # Compute the L1 cost between boxes
        tgt_bbox = bbox_normalize(tgt_bbox, w=size[1], h=size[0])
        # p indicates using L1Norm, Shape: (N*num_queries, len(tgt_ids))
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # out_bbox->cxcywh

        tgt_bbox = box_cxcywh_to_xyxy(tgt_bbox)
        out_bbox = box_cxcywh_to_xyxy(out_bbox)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        Cost = self.lambda_L1 * cost_bbox + self.lambda_cls * cost_class + self.lambda_iou * cost_giou
        Cost = Cost.reshape(N, num_queries, -1)
        C = Cost.detach().cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # multiplier = torch.zeros_like(Cost)
        # for i, c in enumerate(C.split(sizes, -1)):
        #     count = 0  # since we use C.split, the actual idx = count + idx
        #     # we got the row_idx and col_idx of the matched bbox of each image i in that batch
        #     row_idx, col_idx = linear_sum_assignment(c[i])
        #     multiplier[i, row_idx, col_idx + count] = 1
        #     # Use a multiplier matrix with 0 on not matched position and 1 for matched position
        #
        #     # Update count number
        #     count += sizes[i]
        # loss = torch.sum(Cost * multiplier)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return [(i, j) for i, j in indices]


class OWDETRLoss(nn.Module):
    def __init__(self, lambda_cls, lambda_iou, lambda_L1, topK):
        super(OWDETRLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_iou = lambda_iou
        self.lambda_L1 = lambda_L1
        self.matcher = HungarianMatcher(self.lambda_cls, self.lambda_iou, self.lambda_L1)
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.top_unk = topK
        self.num_classes = 91

    def forward(self, img_features, outputs, targets, size):
        # out_prob = outputs["pred_logits"].softmax()  # (N, num_queries, num_cls + 1)
        out_prob = torch.cat([outputs["pred_logits"][..., :-1].softmax(-1),
                              outputs["pred_logits"][..., -1:].sigmoid()], dim=-1)
        out_bbox = outputs["pred_boxes"]  # (N, num_queries, 4)

        queries = torch.arange(out_prob.shape[1])  # Shape of (100)

        device = out_prob.device

        indices_list = self.matcher(outputs, targets, size)
        # we got indices in each image -> [(tensor([14, 16, ...]), tensor([4, 7, ...])), (), (), ...]

        h, w = size
        # samples' size
        upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)  # up sampling to (h, w)

        # initialize loss with 0
        cls_loss = 0
        obj_loss = 0
        iou_loss = 0
        L1_loss = 0

        unk_cls_loss = 0

        mean_img_feat = torch.mean(img_features, 1)
        # res_feat [N, 8, 8] ; img_features after backbone, [N, 2048, 8, 8]
        for i, (query_idx, tgt_idx) in enumerate(indices_list):
            assert query_idx.shape == tgt_idx.shape, "optimized queries and targets number have to be matched"
            # i indicate the image i
            # query_idx -> tensor([14, 16, ...])
            # tgt_idx -> tensor([4, 7, ...])

            other_idx = np.setdiff1d(queries.numpy(), query_idx)  # return not matched indices

            out_matched_cls = out_prob[i, query_idx, 0:-1]  # the class prob of matched queries predicted in image i -> (100, 90)
            out_matched_obj = out_prob[i, query_idx, -1]  # the objectiveness of matched queries predicted in image i
            out_matched_bbox = out_bbox[i, query_idx, :]  # the bbox of matched queries predicted in image i

            out_unmatched_cls = out_prob[i, other_idx, 0:-1]  # the class prob of unmatched queries predicted in image i
            # out_unmatched_obj = out_prob[i, other_idx, -1]  # objectiveness of unmatched queries predicted in image i
            # out_unmatched_bbox = out_bbox[i, other_idx, :]  # the bbox of unmatched queries predicted in image i

            # Suppose 80 boxes unmatched
            upsample_out_bbox = box_cxcywh_to_xyxy(out_bbox[i]) * \
                             torch.tensor([w, h, w, h], dtype=torch.float32).to(device)  # shape -> (100, 4)
            # means_bbox tensor([0,0,...0]) shape of (100)
            means_bbox = torch.zeros(queries.shape[0])

            # (8, 8) -> (1, 1, 8, 8) -> (1, 1, h, w) -> (h, w)
            up_img_feat = upsample(mean_img_feat[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            for j in range(queries.shape[0]):
                if j in other_idx:
                    xmin, ymin, xmax, ymax = upsample_out_bbox[j, :].long()
                    xmin = max(xmin, 0)
                    ymin = max(ymin, 0)
                    xmax = min(xmax, w)
                    ymax = min(ymax, h)
                    means_bbox[j] = torch.mean(up_img_feat[ymin:ymax, xmin:xmax])
                    if torch.isnan(means_bbox[j]):
                        means_bbox[j] = -10e10
                else:
                    means_bbox[j] = -10e10

            _, unmatched_topK_idx = torch.topk(means_bbox, self.top_unk)

            tgt_img_cls = targets[i]['labels'][tgt_idx].to(device)  # (N, 1)
            tgt_img_bbox = targets[i]['boxes'][tgt_idx, :].to(device)  # (N, 4)

            # Normalize the target bounding box
            tgt_img_bbox = bbox_normalize(tgt_img_bbox, w=size[1], h=size[0])

            L1_loss += torch.dist(out_matched_bbox, tgt_img_bbox)

            # for matched indices' classification and objectiveness loss
            cls_loss += self.ce(out_matched_cls, tgt_img_cls)
            obj_loss += self.bce(out_matched_obj, torch.ones_like(out_matched_obj).to(device))

            # transform target bbox to xyxy format
            out_matched_bbox = box_cxcywh_to_xyxy(out_matched_bbox)
            tgt_img_bbox = box_cxcywh_to_xyxy(tgt_img_bbox)

            iou_loss += torch.sum(1 - box_iou(out_matched_bbox, tgt_img_bbox))

            # unmatched index, their labels must be no class but topk has high objectiveness
            tgt_pseudo_cls = torch.zeros_like(out_unmatched_cls).to(device)  # (80, 90)
            unk_cls_loss += self.mse(out_unmatched_cls, tgt_pseudo_cls)

            # unmatched (not in topk list) Queries objectiveness confidence should be 0
            tgt_pseudo_obj = torch.ones_like(out_prob[i, unmatched_topK_idx, -1]).to(device)
            obj_loss += self.bce(out_prob[i, unmatched_topK_idx, -1], tgt_pseudo_obj)

            # out_unmatched_obj_without = out_prob[i, np.setdiff1d(other_idx, unmatched_topK_idx), -1]
            # obj_loss += self.bce(out_unmatched_obj_without, torch.zeros_like(out_unmatched_obj_without).to(device))

        loss = (cls_loss + unk_cls_loss) * self.lambda_cls + obj_loss * self.lambda_cls \
               + iou_loss * self.lambda_iou + L1_loss * self.lambda_L1

        return loss


if __name__ == '__main__':

    N = 16
    num_queries = 100
    num_class = 91

    preds = {'pred_logits': torch.randn(N, num_queries, num_class + 1),
             'pred_boxes': torch.randn(N, num_queries, 4).sigmoid()}

    targets = []
    for _ in range(N):
        tmp = {}
        num_obj = np.random.randint(2, 20)
        tmp['labels'] = torch.randint(0, 90, [num_obj])
        boxes = torch.randn(num_obj, 4)
        # Add here just to make sure the right bottom corner value is larger than the left top
        boxes[..., 2:] += 1
        tmp['boxes'] = boxes.sigmoid()
        targets.append(tmp)

    # loss_fn = DETRLoss(1, 1, 1)
    # print(loss_fn(preds, targets, size=(500, 800)))
