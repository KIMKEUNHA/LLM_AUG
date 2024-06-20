DATA_SETS="origin aug fix origin+aug origin+fix"
for l in $DATA_SETS
    do
        CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset $l
    done