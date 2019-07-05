#!/bin/bash
srun -p AD -n1 --gres=gpu:8 \
    python -u main.py data
