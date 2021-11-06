#!/usr/bin/env bash

#NAME="aclImdbSimple_pretrained_mixed"
NAME="agnews_pretrained"
OUT="temp/$NAME"
#DATASET="aclImdb_tok"
DATASET="agnews"
'''
For Elec, AGNews and DBpedia, use lines below to download pretrained embeddings
'''
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
mv crawl-300d-2M.vec pretrained_embeddings/
 # Preprocess
# python3 scripts/preprocessing.py --train "raw_datasets/aclImdb_tok/train.txt" \
#               --valid "raw_datasets/aclImdb_tok/test.txt" \
#               --test "raw_datasets/aclImdb_tok/test.txt" \
#               --unlabel "raw_datasets/aclImdb_tok/test.txt" \
#               --dataset ${DATASET} \
#               -o ${OUT}/
 python3 scripts/preprocessing.py --train "raw_datasets/agnews/train.txt" \
               --valid "raw_datasets/agnews/test.txt" \
               --test "raw_datasets/agnews/test.txt" \
               --unlabel "raw_datasets/agnews/unlabel.txt" \
               --dataset ${DATASET} \
               -o ${OUT}/
 echo "loading pretrained vectors..."
# python3 scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
#             --data_name "aclImdb_tok" \
#             --pretrained_embeddings "pretrained_embeddings/vectors_aclImdb.txt"
 python3 scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
             --data_name "agnews" \
             --pretrained_embeddings "pretrained_embeddings/crawl-300d-2M.vec"

 echo "preprocessing finished"
#
#export PYTHONPATH="$PWD"
#python3 scripts/main.py --data_folder=${OUT} \
#              --dataset_name=${DATASET} \
#              --random_seed=10 \
#              --words_per_batch=10000 \
#              --num_epochs=10 \
#              --eval_freq=100
