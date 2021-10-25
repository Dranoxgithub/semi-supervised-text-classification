from gensim.models import KeyedVectors
import numpy as np
import argparse
import pickle
import io
import os


def load_pretrained_embeddings(embedding_dir, word2idx_dict, rand_init_range=0.0):
    embeddings = KeyedVectors.load_word2vec_format(embedding_dir, binary=False, unicode_errors='ignore')
    num = 0
    output_embedding = np.zeros((len(word2idx_dict), embeddings.vector_size))
    for i in range(len(word2idx_dict)):
        word = word2idx_dict[i]
        try:
            word_emb = embeddings.vectors[embeddings.key_to_index[word]]
            num += 1
        except KeyError:
            word_emb = np.random.uniform(-rand_init_range, rand_init_range, embeddings.vector_size)
        output_embedding[i,:] = word_emb
    print(f'{num/float(len(word2idx_dict))*100:.2f} % of words are in pretrained embedding')
    return output_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='loads pretrained word embeddings')
    parser.add_argument('--preprocessed_dir', required=True, help='path to preprocessed data')
    parser.add_argument('--data_name', required=True, help='name of the preprocessed file')
    parser.add_argument('--pretrained_embeddings', required=True, help='path to pretrained embeddings')
    args = parser.parse_args()

    with io.open(os.path.join(args.preprocessed_dir, args.data_name + '.vocab.pkl'), 'rb') as f:
        id2w = pickle.load(f)

    word_embeddings = load_pretrained_embeddings(args.pretrained_embeddings, id2w, rand_init_range=0.25)
    np.save(os.path.join(args.preprocessed_dir, args.data_name + '.word_vectors.npy'), word_embeddings)
