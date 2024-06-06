#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
python script to evaluate the SSC model. Adapted from PALNet by jieli_cn@163.com
"""


import os
import torch
import argparse
import datetime

from dataloaders import make_data_loader
from models import make_model
from ssc_network.train import validate_on_dataset


parser = argparse.ArgumentParser(description='PyTorch SSC Training')
parser.add_argument('--dataset', type=str, default='nyu', choices=['nyu', 'nyucad', 'debug'],
                    help='dataset name (default: nyu)')
parser.add_argument('--model', type=str, default='sscnet', choices=['sscnet'],
                    help='model name (default: sscnet)')
parser.add_argument('--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


global args
args = parser.parse_args()


def main():
    # ---- Check CUDA
    if torch.cuda.is_available():
        print("Great, You have {} CUDA device!".format(torch.cuda.device_count()))
    else:
        print("Sorry, You DO NOT have a CUDA device!")

    time_start = datetime.datetime.now()
    test()
    print('Testing finished in: {}'.format(
        datetime.datetime.now() - time_start))


def test():
    # ---- create model ---------- ---------- ---------- ---------- ----------#
    net = make_model(args.model, num_classes=12).cuda()

    # ---- load pretrained model --------- ---------- ----------#
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        cp_states = torch.load(args.resume)
        net.load_state_dict(cp_states['state_dict'], strict=True)
    else:
        raise Exception("=> NO checkpoint found at '{}'".format(args.resume))

    # ---- Data loader
    _, val_loader = make_data_loader(args)

    torch.cuda.empty_cache()

    # ---- Evaluation
    v_prec, v_recall, v_iou, v_acc, v_ssc_iou, v_mean_iou, occupancy_calibration = validate_on_dataset(
        net, val_loader)
    print('Validate with TSDF: p {:.1f}, r {:.1f}, IoU {:.1f}'.format(
        v_prec*100.0, v_recall*100.0, v_iou*100.0))
    print('pixel-acc {:.4f}, mean IoU {:.1f}, SSC IoU:{}'.format(v_acc *
          100.0, v_mean_iou*100.0, v_ssc_iou*100.0))
    print(
        f"Occupancy calibration binary: {occupancy_calibration[0]:.3f} free, {occupancy_calibration[1]:.3f} occ.")
    print("Occupancy calibration semantic:")
    print(occupancy_calibration[2:])


if __name__ == '__main__':
    main()
