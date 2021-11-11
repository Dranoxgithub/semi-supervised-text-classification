#!/usr/bin/env bash

NAME="imdb_pretrained"
OUT="temp/$NAME"
DATASET="imdb"

# for the first time, run the following line to load dataset
python3 scripts/load_imdb.py

# Preprocess
python3 scripts/preprocessing.py --train "raw_datasets/imdb/train.txt" \
               --valid "raw_datasets/imdb/test.txt" \
               --test "raw_datasets/imdb/test.txt" \
               --unlabel "raw_datasets/imdb/unlabel.txt" \
               --dataset ${DATASET} \
               -o ${OUT}/
echo "loading pretrained vectors..."
python3 scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
             --data_name "imdb" \
             --pretrained_embeddings "pretrained_embeddings/imdb_pretrained.txt"
echo "preprocessing finished"

export PYTHONPATH="$PWD"
python3 scripts/main.py --data_folder=${OUT} \
              --dataset_name=${DATASET} \
              --random_seed=10 \
              --words_per_batch=10000 \
              --num_epochs=10 \
              --eval_freq=100 \
              --use_AT \
              --use_CE \
              --use_EM \
              --use_VAT \
              --num_labels=4
