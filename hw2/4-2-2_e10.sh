#! /bin/bash

python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg --discount 0.99 --exp_name rtg_na_l2_16_1000_relu_d99_lr001 --learning_rate 1e-2 --n_layers 2 --size 16
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg --discount 0.99 --exp_name rtg_na_l1_32_1000_relu_d99_lr001 --learning_rate 1e-2 --n_layers 1 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg --discount 0.99 --exp_name rtg_na_l2_32_1000_relu_d99_lr001 --learning_rate 1e-2 --n_layers 2 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg --discount 0.99 --exp_name rtg_na_l1_64_1000_relu_d99_lr001 --learning_rate 1e-2 --n_layers 1 --size 64
