#!/usr/bin/env bash

NAME="aclImdbSimple_pretrained_mixed"
OUT="../temp/$NAME"

if [ -d "$OUT" ]; then
  echo "dir: $OUT   exists, assuming preprocessing complete, if not delete temp and rerun"
else
  mkdir -p ${OUT}
  # Preprocess
  python3 ../scripts/preprocessing.py --train "../raw_datasets/aclImdb_tok/train.txt" \
                --valid "../raw_datasets/aclImdb_tok/test.txt" \
                --test "../raw_datasets/aclImdb_tok/test.txt" \
                --unlabel "../raw_datasets/aclImdb_tok/test.txt" \
                --dataset "aclImdb_tok" \
                -o ${OUT}/
  echo "loading pretrained vectors..."
  python3 ../scripts/load_pretrained_embeddings.py --preprocessed_dir ${OUT} \
              --data_name "aclImdb_tok" \
              --pretrained_embeddings "../pretrained_embeddings/vectors_aclImdb.txt"
fi
echo "preprocessing finished"



