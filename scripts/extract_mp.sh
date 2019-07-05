#!/bin/bash
part=$1
node=$2
srun -p AD -w SH-IDC1-10-5-30-$node -n1 --gres=gpu:8 \
    python -u examples/extract.py \
        data/openimages/images \
        data/openimages/lists/train_${part}.txt \
        -a efficientnet-b5 \
        -b 256 --world-size 1 --rank 0 \
        --dist-url "tcp://10.5.30.$node:23456" \
        --seed 0 --multiprocessing-distributed \
        --output data/openimages/features/feat9m_${part}.npy
