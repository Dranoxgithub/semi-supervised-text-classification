#!/usr/bin/env bash

NAME="gap"
OUT="temp/$NAME"
DATASET="gap_processed"

# get pretrained embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
mv crawl-300d-2M.vec pretrained_embeddings/

# Preprocess
python3 scripts/preprocessing.py --train "raw_datasets/gap_processed/dev.txt" \
             --valid "raw_datasets/gap_processed/valid.txt" \
             --test "raw_datasets/gap_processed/test.txt" \
             --unlabel "raw_datasets/gap_processed/dev.txt" \
             --dataset ${DATASET} \
             -o ${OUT}/
echo "loading pretrained vectors..."
python3 scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
            --data_name "gap_processed" \
            --pretrained_embeddings "pretrained_embeddings/crawl-300d-2M.vec"
echo "preprocessing finished"
export PYTHONPATH="$PWD"
python3 scripts/main.py --data_folder=${OUT} \
              --dataset_name=${DATASET} \
              --random_seed=10 \
              --words_per_batch=5000 \
              --num_epochs=100 \
              --eval_freq=100 \
              --num_labels=2 \
              --logging_freq=50 \
              --ml_loss_weight=1 \
              --at_loss_weight=1 \
              --vat_loss_weight=1 \
              --em_loss_weight=1 \
              --total_loss_weight=1 \
              --enable_logging \
              --use_CE \
              --use_AT \
              --use_EM \
              --use_VAT

