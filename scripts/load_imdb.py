from datasets import load_dataset
from gensim.models import Word2Vec
import os

# create directory if not exist
path = "../raw_datasets/imdb"
if not os.path.exists(path):
    os.mkdir(path)

# load imdb dataset from huggingface
imdb = load_dataset('imdb')
train_text = imdb['train']['text']
train_label = imdb['train']['label']
test_text = imdb['test']['text']
test_label = imdb['test']['label']
unlabel_text = imdb['unsupervised']['text']
unlabel_label = imdb['unsupervised']['text']
train_len = 25000
unlabel_len = 50000

# defining paths

train_path = path + "/train.txt"
test_path = path + "/test.txt"
unlabel_path = path + "/unlabel.txt"
corpus_path = path + "/imdb_corpus.txt"
embedding_path = "../pretrained_embeddings/imdb_pretrained.txt"

# save datasets
with open(train_path, 'w') as file:
    for i in range(train_len):
        temp = str(train_label[i]) + "\t  " + "%s\n" % train_text[i]
        file.write(temp.lower())

with open(test_path, 'w') as file:
    for i in range(train_len):
        temp = str(test_label[i]) + "\t  " + "%s\n" % test_text[i]
        file.write(temp.lower())

with open(unlabel_path, 'w') as file:
    for i in range(unlabel_len):
        temp = "-1\t  " + "%s\n" % unlabel_text[i]
        file.write(temp.lower())

# save corpus file
corpus = train_text + test_text + unlabel_text
with open(corpus_path, 'w') as file:
    for item in corpus:
        file.write(item.lower())

# create word embedding vectors using corpus file above
model = Word2Vec(corpus_file=corpus_path, vector_size=300)
word_vector = model.wv
del model
word_vector.save_word2vec_format(embedding_path)