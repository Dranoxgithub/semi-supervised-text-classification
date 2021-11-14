#!/usr/bin/env bash

NAME="arxiv_pretrained_mixed"
OUT="temp/$NAME"
DATASET="arxiv"

# run the following line to get raw data and load embeddings
# need to manually upload kaggle.json to current directory first; the following lines will take care of it
pip install kaggle
mkdir ~/.kaggle
mv ./kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
mkdir raw_datasets/arxiv
kaggle datasets download Cornell-University/arxiv
unzip arxiv.zip
mv arxiv-metadata-oai-snapshot.json raw_datasets/arxiv
# the following line will print out unique number of labels; should be 11128 since random seed is fixed
python3 scripts/load_arxiv.py

# Preprocess
python3 scripts/preprocessing.py --train "raw_datasets/arxiv/train.txt" \
               --valid "raw_datasets/arxiv/test.txt" \
               --test "raw_datasets/arxiv/test.txt" \
               --unlabel "raw_datasets/arxiv/unlabel.txt" \
               --dataset ${DATASET} \
               -o ${OUT}/
echo "loading pretrained vectors..."
python3 scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
             --data_name "arxiv" \
             --pretrained_embeddings "pretrained_embeddings/arxiv_pretrained.txt"
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
