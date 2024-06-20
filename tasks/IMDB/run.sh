#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset "origin" --model_load_ckpt_pth "src/checkpoints/IMDB_fix-ep3-lr5e-05-bsz16/ckpt-epoch=001-val_loss=0.40180.ckpt"
