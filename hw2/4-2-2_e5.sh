#! /bin/bash

python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 3 --discount 0.90 -rtg --exp_name rtg_na_l1_32_5000_relu_d90 --n_layers 1 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 3 --discount 0.80 -rtg --exp_name rtg_na_l1_32_5000_relu_d80 --n_layers 1 --size 32