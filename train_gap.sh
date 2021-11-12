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
echo "loading pretrained vectors..."