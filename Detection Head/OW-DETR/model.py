import yaml
import json

from torch.optim import Adam
from torch.utils.data import DataLoader

from backbone import Backbone
from box_ops import box_cxcywh_to_xyxy
from config import *
from dataset import CoCoDataset, detection_collate
from detr import DETR
from owdetr_loss import OWDETRLoss
from utils import nested_tensor_from_tensor_list
from utils import save_checkpoint, load_checkpoint


class Model(object):
    def __init__(self, dataset_config, model_config):
        self.dataset_config = self._parse_configs(dataset_config)
        self.model_config = self._parse_configs(model_config)

        self.loader = self.build_dataset()
        self.backbone, self.model = self.build()

    @staticmethod
    def _parse_configs(path):
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def write_json(output):
        output['box'] = output['box'].detach().cpu().numpy().tolist()
        output['cls'] = output['cls'].detach().cpu().numpy().tolist()
        json_string = json.dumps(output)
        with open('outputs/{}.json'.format(output['img_id']), 'w') as outfile:
            outfile.write(json_string)

    def build_dataset(self):
        coco_train = CoCoDataset(**self.dataset_config.get('set_param'), transform=transform)
        loader = DataLoader(coco_train, collate_fn=detection_collate, **self.dataset_config.get('loader_param'))
        return loader

    def build(self):
        backbone = Backbone(**self.model_config.get('backbone_param'))
        model = DETR(**self.model_config.get('detr_param')).to(self.model_config.get('device'))
        return backbone, model

    def train(self):
        optim = Adam(self.model.parameters(), lr=self.model_config.get('lr'))
        criterion = OWDETRLoss(**self.model_config.get('loss'))
        # img_ids, tensor_list, targets = next(iter(self.loader))
        for epoch in range(self.model_config.get('num_epoch')):
            for img_ids, tensor_list, targets in self.loader:
                # tensor list means a list of img tensors
                if self.model_config.get('save_model') and epoch % 2 == 0 and epoch != 0:
                    checkpoint = {'model': self.model.state_dict(), 'optim': optim.state_dict()}
                    save_checkpoint(checkpoint, epoch)

                nested_list = nested_tensor_from_tensor_list(tensor_list)
                h, w = nested_list.mask.shape[1], nested_list.mask.shape[2]

                img_feature = self.backbone(nested_list)['0'].tensors  # torch.Size([16, 2048, 20, 20])
                img_feature = img_feature.to(device)
                img_feature = img_feature.sigmoid()
                preds = self.model(img_feature)
                # preds = model(img_feature)
                loss = criterion(img_feature, preds, targets, size=(h, w))

                print(loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()

    def eval(self):
        img_ids, tensor_list, targets = next(iter(self.loader))
        nested_list = nested_tensor_from_tensor_list(tensor_list)
        img_feature = self.backbone(nested_list)['0'].tensors  # torch.Size([16, 2048, 20, 20])
        img_feature = img_feature.to(self.model_config.get('device'))
        img_feature = img_feature.sigmoid()
        preds = self.model(img_feature)
        pred_boxes = preds['pred_boxes']
        boxes = box_cxcywh_to_xyxy(pred_boxes)
        preds_logits = torch.cat([preds["pred_logits"][..., :-1].softmax(-1),
                                  preds["pred_logits"][..., -1:].sigmoid()], dim=-1)
        output = dict()
        print('getting results...')
        for img_id, box, logit in zip(img_ids, boxes, preds_logits):
            output['img_id'] = img_id
            # convert to numpy for adapting JSON format
            output['box'] = box[logit[..., -1] > 0.80]
            output['cls'] = torch.argmax(logit[..., :-1][logit[..., -1] > 0.80], dim=-1)
            self.write_json(output)


if __name__ == '__main__':
    model = Model(dataset_config='configs/dataset_config.yml', model_config='configs/model_config.yml')
    model.train()
    model.eval()
