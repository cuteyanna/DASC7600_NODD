import torch
from collections import Counter
from box_ops import box_iou


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_class=90):
    """
    :param pred_boxes: predict boxes -> list: [[train_idx, class_pred, prob, x1, y1, x2, y2], [], [], ...]
    :param true_boxes: target boxes -> list: [[train_idx, class, x1, y1, x2, y2], [], [], ...]
    :param iou_threshold: > iou_threshold -> TP, < iou_threshold -> FP
    :param num_class: number of classes
    :return: mAP
    """

    average_precision = []
    epsilon = 1e-6

    # calculate AP for each class
    for cls in range(num_class):
        detections = []
        ground_truths = []

        # select specific class in all pred boxes
        for detection in pred_boxes:
            if detection[1] == cls:
                detections.append(detection)

        # select coordinate true boxes in that specific class
        for true_box in true_boxes:
            if true_box[1] == cls:
                ground_truths.append(true_box)

        # Counter example: img1: 3 bboxes, img2: 5 bboxes -> {1: 3, 2: 5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # amount_bboxes -> {1: torch.tensor([0, 0, 0]), 2: torch.tensor([0 ,0 ,0 ,0 ,0])}

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_imgs = [bbox for bbox in ground_truths if bbox[0] == detection[0]]  # from the same image

            best_iou = 0
            best_gt_idx = None

            for idx, gt in enumerate(ground_truth_imgs):
                iou = box_iou(detection, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold and best_gt_idx is not None:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection] = 1

            else:
                FP[detection] = 1

            # [1, 1, 0, 0, 1] -> [1, 2, 2, 2, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        # To calculate the area of PR curve, we need to add a point of (0, 1)
        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = torch.cat([torch.tensor([0])], recalls)
        average_precision.append(torch.trapz(precisions, recalls))

    return sum(average_precision) / len(average_precision)









