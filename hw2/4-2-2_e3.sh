#! /bin/bash

python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg --exp_name rtg_na_l1_32_5000_relu --n_layers 1 --size 32