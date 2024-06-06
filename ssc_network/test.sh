#!/bin/bash

python ./test.py \
--model='SSCNet' \
--dataset=nyu \
--batch_size=4 \
--resume='pretrained_models/weights/SSCNet.pth.tar' 2>&1 |tee SSCNet_NYU_test.log


