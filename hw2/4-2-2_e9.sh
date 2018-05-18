#! /bin/bash

python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg --exp_name rtg_na_l2_16_1000_relu_lr001 --learning_rate 1e-2 --n_layers 2 --size 16
