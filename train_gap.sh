#!/usr/bin/env bash

NAME="gap"
OUT="temp/$NAME"
DATASET="gap_processed"

# Preprocess
python3 scripts/preprocessing.py --train "raw_datasets/gap_processed/dev.txt" \
              --valid "raw_datasets/gap_processed/valid.txt" \
              --test "raw_datasets/gap_processed/test.txt" \
              --unlabel "raw_datasets/gap_processed/dev.txt" \
              --dataset ${DATASET} \
              -o ${OUT}/
#echo "loading pretrained vectors..."
#python3 scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
#             --data_name "gap_processed" \
#             --pretrained_embeddings "pretrained_embeddings/crawl-300d-2M.vec"
#echo "preprocessing finished"
export PYTHONPATH="$PWD"
python3 scripts/main.py --data_folder=${OUT} \
              --dataset_name=${DATASET} \
              --random_seed=10 \
              --words_per_batch=5000 \
              --num_epochs=100 \
              --eval_freq=100 \
              --num_labels=2 \
              --use_CE \
              --enable_logging \
              --logging_freq=10 \
              --use_AT \
              --use_EM \
              --use_VAT

