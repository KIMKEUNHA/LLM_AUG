#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python src/main.py --do_test_only True --model_load_ckpt_pth "src/checkpoints/IMDB_origin-ep3-lr5e-05-bsz32/ckpt-epoch=001-val_loss=0.15883.ckpt"
