#! /bin/bash

python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 1 -rtg --exp_name rtg_na_l1_32_1000 --n_layers 1 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 10000 -e 1 -rtg --exp_name rtg_na_l1_32_10000 --n_layers 1 --size 32