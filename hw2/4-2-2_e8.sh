#! /bin/bash

python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 5 --discount 0.99 -lr 0.01 -rtg --exp_name rtg_na_l1_32_5000_relu_d99_lr001 --n_layers 1 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 5 --discount 0.99 -lr 0.01 -rtg --exp_name rtg_na_l2_32_5000_relu_d99_lr001 --n_layers 2 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 5 --discount 0.99 -lr 0.01 -rtg --exp_name rtg_na_l1_64_5000_relu_d99_lr001 --n_layers 1 --size 64
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 5 --discount 0.99 -lr 0.01 -rtg --exp_name rtg_na_l2_16_5000_relu_d99_lr001 --n_layers 2 --size 16