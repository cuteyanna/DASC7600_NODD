import json
import os
from collections import defaultdict
import numpy as np
import torch


class Voc_Evaluator:
    def __init__(self, voc_gt, ovthresh=list(range(50, 100, 5))):
        self.lines = []
        self.lines_cls = []

        self.rec = []
        self.prec = []

        self.ovthresh = ovthresh
        self.voc_gt = voc_gt

        self.known_classes = self.voc_gt.CLASS_NAMES

        self.AP = torch.zeros(len(self.known_classes), 1)
        self.all_recs = defaultdict(list)
        self.all_precs = defaultdict(list)


    def update(self, predictions):
        img_id, preds = predictions.popitem()  # we only have one key value pair in the dict
        if preds is not None:
            pred_boxes, pred_labels, pred_scores = [preds[0][k].cpu() for k in ['boxes', 'labels', 'scores']]
            classes = pred_labels.tolist()

            for (xmin, ymin, xmax, ymax), cls, score in zip(pred_boxes.tolist(), classes, pred_scores.tolist()):
                xmin += 1
                ymin += 1
                self.lines.append(f"{img_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")
                self.lines_cls.append(cls)

    def accumulate(self):
        for class_label_ind, class_label in self.voc_gt.CLASS_NAMES.items():
            '''
                self.lines.append(f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")
                self.lines_cls.append(cls)
            '''
            lines_by_class = [l + '\n' for l, c in zip(self.lines, self.lines_cls) if c == class_label_ind]
            if len(lines_by_class) == 0:
                lines_by_class = []
            print(class_label + " has " + str(len(lines_by_class)) + " predictions.")

            ovthresh = 50
            ovthresh_ind, _ = map(self.ovthresh.index, [50, 75])
            self.rec, self.prec, self.AP[class_label_ind, ovthresh_ind] = voc_eval(lines_by_class,
                                                                                   class_label_ind,
                                                                                   self.voc_gt.image_set,
                                                                                   self.voc_gt.annotations,
                                                                                   ovthresh=ovthresh / 100,
                                                                                   )
            self.all_recs[ovthresh].append(self.rec)
            self.all_precs[ovthresh].append(self.prec)

    def summarize(self, fmt='{:.06f}'):
        o50, _ = map(self.ovthresh.index, [50, 75])
        avg_precision_unk = self.compute_avg_precision_at_many_recall_level_for_unk(self.all_precs, self.all_recs)
        print('avg_precision: ' + str(avg_precision_unk))
        mAP = float(self.AP[:-1,...].mean())
        print('detection mAP:', fmt.format(mAP))
        for class_name, ap in zip(self.voc_gt.CLASS_NAMES, self.AP[:, o50].cpu().tolist()):
            print(class_name, fmt.format(ap))

    def compute_avg_precision_at_many_recall_level_for_unk(self, precisions, recalls):
        precs = {}
        for r in range(1, 10):
            r = r / 10
            p = self.compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls,
                                                                     recall_level=r)
            precs[r] = p
        return precs

    def compute_avg_precision_at_a_recall_level_for_unk(self, precisions, recalls, recall_level=0.5):
        precs = {}
        for iou, recall in recalls.items():
            prec = {}
            avg = 0
            for cls_id, rec in enumerate(recall):
                if len(rec) > 0:
                    p = precisions[iou][cls_id][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                    avg += p

                else:
                    avg += 0

            prec['known_avg'] = avg / (len(recall) - 1)
            precs[iou] = prec

        return precs


def voc_eval(detpath,
             classname,
             imagepath='voc_data/images/test',
             annopath='voc_data/annotation/test',
             ovthresh=0.5
             ):
    """

    :param detpath: self.lines : list
    :param classname: actually class id here -1 for unk, 0 for person
    :param imagepath: the root dir to test images
    :param annopath: the root dir to test annotation
    :param ovthresh: IOU threshold
    :return:
    """
    imagenames = os.listdir(imagepath)
    # load  annotations

    # recs['2007.jpg'] =
    #                    [{'name':'person'(0),'difficult':0/1,'bbox':[23,33,345,233]},
    #                     {'name':'bird','difficult':0/1,'bbox':[23,33,345,233]}]

    recs = parse_json(imagenames, root_dir=annopath)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # 對於每一張圖片的所有object做循環，有classname的拿出來存在R中
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # R = [{'name':'person'(0),'difficult':0/1,'bbox':[23,33,345,233]},...], all the obj are 'person'
        bbox = np.array([x['bbox'] for x in R])
        # bbox = [[23,33,345,233],[23,33,345,233],...]
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # difficult = [0,0,1,0...]
        det = [False] * len(R)
        # det = [false,false,...]
        # npos = npos + sum(~difficult)
        npos = npos + len(R)
        # npos = the number of objects in this class 'Person'
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    # class_recs['2007.jpg'] = {'bbox':,'difficult':,'det':}

    # read predictions
    lines = detpath

    # lines belong to the same class

    # get image_id, score, bounding box
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    # ['2007.jpg','2008.jpg','2007.jpg'...]
    confidence = np.array([float(x[1]) for x in splitlines])
    # ['0.4','0.5',...]
    if len(splitlines) == 0:
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
    else:
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # .reshape(-1, 4)

    # sort by confidence, 降序排列返回index
    sorted_ind = np.argsort(-confidence)
    # sorted index = [5,4,1,65,...]
    BB = BB[sorted_ind, :]

    image_ids = [image_ids[x] for x in sorted_ind]
    # image_ids = ['2008.jpg','2007.jpg','2007.jpg',...],從大到小

    # go down preds and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        # R = class_recs['2008.jpg'] = {'bbox':[[23,33,345,233],[23,33,345,233],...],'difficult':,'det':} targets
        bb = BB[d, :].astype(float)
        # bb = [23,45,342,454] prediction
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ovmax, jmax = iou(BBGT, bb)
        # ovmax 是最大的iou值，jmax是所對應的index

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    # tp = [1,1,0,1,0,1,0,1,0,1,1]
    tp = np.cumsum(tp)
    # tp = [1,2,2,3,3,4,4,5,5,6,7,...,75], 75 的idx 130
    rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    # avoid divide by zero in case the first detection matches a difficult

    tp_plus_fp_closed_set = tp + fp

    return rec, prec, ap


def parse_json(imagenames, root_dir='voc_data/annotation/test'):
    """
    :param root_dir: 'voc_data/annotation/test'
    :param imagenames: ['2007_0032.jpg', '2007_0033.jpg']
    :return: {'2007_0032.jpg': [{'name': 'person', 'difficult': 0, 'bbox': [234, 345, 245, 456]},
                                {'name': 'person', 'difficult': 0, 'bbox': [234, 345, 245, 456]}]}
    """
    recs = {}
    for imagename in imagenames:
        json_path = os.path.join(root_dir, imagename[:-4] + '.json')
        objects = []
        with open(json_path) as f:
            img_rec = json.load(f)

        for obj in img_rec.get('annotations'):
            objects.append({'name': obj.get('category_id'), 'bbox': obj.get('bbox'), 'difficult': obj.get('difficult')})

        recs[img_rec.get('images')['file_name']] = objects

    return recs


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def iou(BBGT, bb):
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return ovmax, jmax



