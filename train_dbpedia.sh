#!/usr/bin/env bash

NAME="dbpedia_pretrained_mixed"
OUT="temp/$NAME"
DATASET="dbpedia"
#For Elec, AGNews and DBpedia, use lines below to download pretrained embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
mv crawl-300d-2M.vec pretrained_embeddings/

#unzip tar.gz file
tar -xvf raw_datasets/dbpedia/train.tar.gz
mv train.txt raw_datasets/dbpedia
tar -xvf raw_datasets/dbpedia/test.tar.gz
mv test.txt raw_datasets/dbpedia

# Preprocess
python3 scripts/preprocessing.py --train "raw_datasets/dbpedia/train.txt" \
              --valid "raw_datasets/dbpedia/test.txt" \
              --test "raw_datasets/dbpedia/test.txt" \
              --unlabel "raw_datasets/dbpedia/train.txt" \
              --dataset ${DATASET} \
              -o ${OUT}/
echo "loading pretrained vectors..."
python3 scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
            --data_name "dbpedia" \
            --pretrained_embeddings "pretrained_embeddings/crawl-300d-2M.vec"
echo "preprocessing finished"

export PYTHONPATH="$PWD"
python3 scripts/main.py --data_folder=${OUT} \
              --dataset_name=${DATASET} \
              --random_seed=50 \
              --words_per_batch=5000 \
              --num_epochs=50 \
              --eval_freq=100 \
              --num_labels=14 \
              --enable_logging \
              --logging_freq=10 \
              --use_CE \
              --use_AT \
              --use_EM \
              --use_VAT
