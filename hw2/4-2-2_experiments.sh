#! /bin/bash

python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg --exp_name rtg_na_l1_32 --n_layers 1 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg --exp_name rtg_na_l1_64 --n_layers 1 --size 64
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg --exp_name rtg_na_l2_32 --n_layers 2 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg --exp_name rtg_na_l2_64 --n_layers 2 --size 64
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg --exp_name rtg_na_l3_32 --n_layers 3 --size 32
python train_pg.py InvertedPendulum-v2 -n 100 -b 5000 -e 1 -rtg --exp_name rtg_na_l3_64 --n_layers 3 --size 64