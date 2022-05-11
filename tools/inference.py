# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import cv2
import torch

import argparse

from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
#from nanodet.util import load_model_weight
import datetime

from pycocotools.coco import COCO

from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    mkdir,
load_model_weight,
)


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = [height]
        img_info["width"] = [width]
        img_info["id"] = [1]
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline("dataset", meta, self.cfg.data.val.input_size)
        meta["img"] = (
            torch.from_numpy(meta["img"].transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        self.model.head.show_result(
            meta["raw_img"], dets, class_names, score_thres=score_thres, show=True
        )
        print("viz time: {:.3f}s".format(time.time() - time1))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    load_config(cfg, args.config)

    local_rank = -1
    cfg.defrost()
    timestr = datetime.datetime.now().__format__("%Y%m%d%H%M%S")
    cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    mkdir(local_rank, cfg.save_dir)
    logger = NanoDetLightningLogger(cfg.save_dir)

    predict = Predictor(cfg, args.model, logger)

    img_path = '/root/deng/dataset/coco/mine/val/person_cat_dog/'
    # ann_path = 'D:\\deng\\dataset\\coco\\2017\\annotations\\' + ann_file
    # img_path = 'D:\\deng\\dataset\\coco\\2017\\val2017\\'
    ann_path = '/root/deng/dataset/coco/mine/val/person_cat_dog_val.json'
    coco_api = COCO(ann_path)

    imgIds = coco_api.getImgIds()

    for imgId in imgIds:
        imgInfo = coco_api.loadImgs(imgId)[0]
        img_file = img_path + imgInfo["file_name"]
        results = predict.inference(img_file)
        print(results)
        break