#!/usr/bin/env bash

NAME="aclImdbSimple_pretrained_mixed"
OUT="temp/$NAME"
DATASET="aclImdb_tok"

# Preprocess
#python3 scripts/preprocessing.py --train "raw_datasets/aclImdb_tok/train.txt" \
#              --valid "raw_datasets/aclImdb_tok/test.txt" \
#              --test "raw_datasets/aclImdb_tok/test.txt" \
#              --unlabel "raw_datasets/aclImdb_tok/test.txt" \
#              --dataset ${DATASET} \
#              -o ${OUT}/
#echo "loading pretrained vectors..."
#python3 scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
#            --data_name "aclImdb_tok" \
#            --pretrained_embeddings "pretrained_embeddings/vectors_aclImdb.txt"
#echo "preprocessing finished"


export PYTHONPATH="$PWD"
python3 scripts/main.py --data_folder=${OUT} \
              --dataset_name=${DATASET} \
              --random_seed=50 \
              --words_per_batch=5000 \
              --num_epochs=50 \
              --eval_freq=100 \
              --num_labels=2 \
              --enable_logging \
              --logging_freq=10 \
              --use_CE \
              --use_AT \
              --use_EM \
              --use_VAT