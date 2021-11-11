import argparse as ap
import nltk
from nltk import word_tokenize
import collections
import io
import re
import numpy as np
import os
import pickle
import progressbar
import torch

def parse_args():
    parser = ap.ArgumentParser(description="Parse the command line arguments")
    parser.add_argument('--dataset',type=str, dest="dataset", required=True, help="Name of the dataset")
    parser.add_argument('--train',type=str, dest="train", required=True, help="Training dataset file path")
    parser.add_argument('--valid',type=str, dest="valid", required=True, help="Valid dataset file path")
    parser.add_argument('--test',type=str, dest="test", required=True, help="Test dataset file path")
    parser.add_argument('--unlabel',type=str, dest="unlabel", required=True, help="Unlabeled dataset file path")

    parser.add_argument('--vocab_size', type=int, dest="vocab_size", default=80000, help='Vocabulary size of source language')
        
    parser.add_argument('--max_len', type=int, dest="max_len", default=1000, help='Maximum sequence length')
    parser.add_argument('--output_dir', '-o', type=str, dest="output_dir", default='temp', help='Output directory')

    return parser.parse_args()

def read_file(path):
    n_lines = len(open(path).readlines(  ))
    bar = progressbar.ProgressBar()
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for line in bar(f, max_value=n_lines):
            # only split at the first tab
            line_list = line.strip().split('\t', 1)
            assert len(line_list) == 2

            label = line_list[0]
            line_list = line_list[1].lower().replace('\u2019', "'")
            words = word_tokenize(line_list)  
            yield label, words

# get the most frequent max_vocab_size number of words from each file in file_path_list
def count_vocabs_and_labels(file_path_list, max_vocab_size=40000):
    all_vocabs = set()
    all_labels = set()
    for file_path in file_path_list:
        counts = collections.Counter()
        for label, words in read_file(file_path):
            for word in words:
                counts[word] += 1
            all_labels.add(label)
        for (word, _) in counts.most_common(max_vocab_size):
            all_vocabs.add(word)
    return all_vocabs, all_labels

def create_dataset(path, w2id, label2id, max_len, pad_label_as_negative_one = False):
    dataset = []
    num_tokens = 0
    num_unknown_padding = 0
    for label, words in read_file(path):
        tokenized_words = [w2id.get(word, w2id.get('<unk>')) for word in words]
        assert len(tokenized_words) > 0
        if len(tokenized_words) >= max_len: continue
        num_tokens += len(tokenized_words)
        num_unknown_padding += tokenized_words.count(w2id.get('<unk>'))

        label_index = -1 if pad_label_as_negative_one else label2id[label]
        dataset.append((label_index, tokenized_words))
    print(f'# of tokens in {path}: {num_tokens}')
    print(f'number of unknown in {path}: {num_unknown_padding}')
    return dataset

if __name__ == "__main__":
    nltk.download('punkt')
    args = parse_args()

    # get the most common word from train, valid and unlabel 
    vocabs_set, labels_set = count_vocabs_and_labels([args.train, args.valid, args.unlabel], args.vocab_size)
    labels_list = sorted(list(labels_set))

    vocabs_list = ['<pad>', '<eos>', '<unk>', '<bos>'] + list(vocabs_set)
    w2id = {word: index for index, word in enumerate(vocabs_list)}
    # encode all labels
    label2id = {l: index for index, l in enumerate(labels_list)}
    print("Finish getting all vocabs and labels.")
    
    id2w = {i: w for w, i in w2id.items()}

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pickle.dump(id2w, open(args.output_dir + args.dataset + '.vocab.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    id2label = {i: l for l, i in label2id.items()}
    pickle.dump(id2label, open(args.output_dir + args.dataset + '.label.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    train_dataset = create_dataset(args.train, w2id, label2id, args.max_len)
    valid_dataset = create_dataset(args.valid, w2id, label2id, float('inf'))
    test_dataset = create_dataset(args.test, w2id, label2id, float('inf'))
    unlabel_dataset = create_dataset(args.unlabel, w2id, label2id, float('inf'), True)

    # save all tokenized datasets    
    pickle.dump(train_dataset, open(args.output_dir + args.dataset + '.train.pkl', "wb"))
    pickle.dump(valid_dataset, open(args.output_dir + args.dataset + '.valid.pkl', "wb"))
    pickle.dump(test_dataset, open(args.output_dir + args.dataset + '.test.pkl', "wb"))
    pickle.dump(unlabel_dataset, open(args.output_dir + args.dataset + '.unlabel.pkl', "wb"))
