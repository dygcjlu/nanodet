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

import argparse
import datetime
import os
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import normalize
import cv2

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    mkdir,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="val", help="task to run, test or val"
    )
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    args = parser.parse_args()
    return args

def evalFromLocal(args):
    load_config(cfg, args.config)
    local_rank = -1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    cfg.defrost()
    timestr = datetime.datetime.now().__format__("%Y%m%d%H%M%S")
    cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    mkdir(local_rank, cfg.save_dir)
    logger = NanoDetLightningLogger(cfg.save_dir)

    assert args.task in ["val", "test"]
    cfg.update({"test_mode": args.task})

    logger.info("Setting up data...")
    val_dataset = build_dataset(cfg.data.val, args.task)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    #json_path = '/root/deng/nanodet/mine/nanodet/workspace/ncnn_result/result.json'
    json_path = '/root/deng/pet/ezqdk/v3/ezqdk/projects/reference/nanodet/results.json'
    eval_results = evaluator.evalFromLocal( json_path)
    print(eval_results)
    """
    txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
    with open(txt_path, "a") as f:
        for k, v in eval_results.items():
            f.write("{}: {}\n".format(k, v))"""


def main(args):
    load_config(cfg, args.config)
    local_rank = -1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    cfg.defrost()
    timestr = datetime.datetime.now().__format__("%Y%m%d%H%M%S")
    cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    mkdir(local_rank, cfg.save_dir)
    logger = NanoDetLightningLogger(cfg.save_dir)

    assert args.task in ["val", "test"]
    cfg.update({"test_mode": args.task})

    logger.info("Setting up data...")
    val_dataset = build_dataset(cfg.data.val, args.task)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)

    ckpt = torch.load(args.model)
    if "pytorch-lightning_version" not in ckpt:
        warnings.warn(
            "Warning! Old .pth checkpoint is deprecated. "
            "Convert the checkpoint with tools/convert_old_checkpoint.py "
        )
        ckpt = convert_old_model(ckpt)
    task.load_state_dict(ckpt["state_dict"])

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        gpus=cfg.device.gpu_ids,
        accelerator="ddp",
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        logger=logger,
    )
    logger.info("Starting testing...")
    trainer.test(task, val_dataloader)

def DeConvTest():
    pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    unpool = nn.MaxUnpool2d(2, stride=2)
    input = torch.tensor([[[[1., 2., 3., 4.],
                                 [5., 6., 7., 8.],
                                 [9., 10., 11., 12.],
                                 [13., 14., 15., 16.]]]])
    output, indices = pool(input)
    output = unpool(output, indices)
    print(output.size())

    input = torch.randn(1, 16, 20, 20)
    downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
    #upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
    upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=0, dilation=1)
    downsample2 = nn.Conv2d(16, 16, 3, stride=1, padding=1, dilation=2)

    unpool = nn.MaxUnpool2d(2, stride=2)
    #upsample = nn.ConvTranspose2d(16, 16, 2, stride=2)
    h = downsample(input)
    print(h.size())
    #torch.Size([1, 16, 20, 20])
    input2 = torch.randn(1, 16, 20, 20)
    #output = upsample(input2,  output_size=input.size())
    #output = upsample(input2)
    #output = downsample2(output)
    output = unpool(input2)
    print(output.size())

def batchNoram():
    before_emb = np.ones((4, 35, 5, 5))
    after_emb =  np.zeros((4, 35, 5, 5))
    before_emb = before_emb.reshape(before_emb.shape[0], -1)
    after_emb = after_emb.reshape(after_emb.shape[0], -1)
    before_emb, after_embsum_sim = normalize(before_emb), normalize(after_emb)
    dot_sim = (before_emb * after_emb)
    sum_sim = dot_sim.sum(axis=1)
    sim = sum_sim.mean()


    return
    print('BatchNorm2d')
    #m = nn.BatchNorm2d(100)
    # Without Learnable Parameters
    m = nn.BatchNorm2d(100, affine=True)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)
    print(m.weight.shape)
    print(m.bias.shape)

    print('GroupNorm')
    input = torch.randn(20, 6, 10, 10)
    # Separate 6 channels into 3 groups
    m = nn.GroupNorm(3, 6)
    # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
    #m = nn.GroupNorm(6, 6)
    # Put all 6 channels into a single group (equivalent with LayerNorm)
    #m = nn.GroupNorm(1, 6)
    # Activating the module
    output = m(input)
    print(m.weight.shape)
    print(m.bias.shape)

def compute_mean_std():
    filepath = r'/root/deng/nanodet/mine/nanodet/workspace/ncnn_result/image/'
    images = os.listdir(filepath)
    #means = [128, 128, 128]
    #stds = [64, 64, 64]
    mean_sum = [0, 0, 0]
    std_sum = [0, 0, 0]
    for image in images:
        file = filepath + image
        img = cv2.imread(file)
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        #np.concatenate([mean, ])
        mean_sum = np.add(mean, mean_sum)
        std_sum = np.add(std, std_sum)

    num = len(images)
    mean = mean_sum * 1.0 / num
    std = std_sum * 1.0 / num
    print(mean)
    print(std)
    """
    img = cv2.imread('/root/deng/nanodet/mine/nanodet/tools/workspace/person_320_320.jpg')
    print(img.shape)
    mean = np.mean(img, axis=(0, 1))
    print(mean)
    std = np.std(img, axis=(0, 1))
    print(std)
    for i in range(3):
        c = img[:, :, i]
        c_mean = np.mean(c)
        print(c_mean)
        c_std = np.std(c)
        print(c_std)
    """

def modify_pretrain_model():
    model_path = '/root/deng/nanodet/mine/nanodet/pretrain/shufflenetv3_0.8x/model_best.pth.tar'
    new_model = '/root/deng/nanodet/mine/nanodet/pretrain/shufflenetv3_0.8x/model_best.pth'
    checkpoint = torch.load(model_path)

    state_dict = checkpoint['state_dict']
    new_sstate_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_sstate_dict[new_key] = value

    torch.save({
            'state_dict': new_sstate_dict,
            }, new_model)
    print("end")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    #batchNoram()
    #evalFromLocal(args)
    #DeConvTest()
    #compute_mean_std()
    #modify_pretrain_model()
