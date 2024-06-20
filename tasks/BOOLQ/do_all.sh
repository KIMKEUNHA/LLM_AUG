DATA_SETS="origin+fix"
for l in $DATA_SETS
    do
        CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset $l --checkpoint_save_top_k 1 --checkpoint_monitor "val_accuracy" --epoch 10
    done
