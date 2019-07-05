#!/bin/bash
srun -p AD -n1 --gres=gpu:8 \
    python -u examples/extract.py \
        data/openimages/images \
        data/openimages/lists/train_less.txt \
        -a efficientnet-b5 \
        -b 32 --world-size 1 --rank 0 \
        --seed 0 --gpu 0 \
        --output data/openimages/features/feat1k_1gpu.npy
