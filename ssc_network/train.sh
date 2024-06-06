#!/usr/bin/env bash

source venv/bin/activate

python ./train.py \
--model='sscnet' \
--dataset=nyu \
--epochs=50 \
--batch_size=1 \
--workers=1 \
--lr=0.01 \
--lr_adj_n=10 \
--lr_adj_rate=0.1 \
--model_name='SSCNet' 2>&1 |tee SSCNet_NYU_train.log

deactivate

