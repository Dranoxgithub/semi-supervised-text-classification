#!/usr/bin/env bash

NAME="aclImdbSimple_pretrained_mixed"
OUT="../temp/$NAME"

# Preprocess
python3 ../scripts/preprocessing.py --train "../raw_datasets/aclImdb_tok/train.txt" \
                --valid "../raw_datasets/aclImdb_tok/test.txt" \
                --test "../raw_datasets/aclImdb_tok/test.txt" \
                --unlabel "../raw_datasets/aclImdb_tok/test.txt" \
                --dataset "aclImdb_tok" \
                -o ${OUT}/processed