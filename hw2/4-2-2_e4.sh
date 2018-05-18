#! /bin/bash

python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 --discount 0.99 -rtg --exp_name rtg_na_l1_32_5000_relu_d99 --n_layers 1 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 --discount 0.95 -rtg --exp_name rtg_na_l1_32_5000_relu_d95 --n_layers 1 --size 32