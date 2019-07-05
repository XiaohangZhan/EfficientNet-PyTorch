#!/bin/bash
node=$1
srun -p AD -w SH-IDC1-10-5-30-$node -n1 --gres=gpu:8 \
    python -u examples/extract.py \
        data/openimages/images \
        data/openimages/lists/validation.txt \
        -a efficientnet-b5 \
        -b 256 --world-size 1 --rank 0 \
        --action "predict" \
        --dist-url "tcp://10.5.30.$node:23456" \
        --seed 0 --multiprocessing-distributed \
        --output data/openimages/predicts/pred_val.npy
