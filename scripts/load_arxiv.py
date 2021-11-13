import datasets
from gensim.models import Word2Vec
import os
import random

path = "../raw_datasets/arxiv"
if not os.path.exists(path):
    os.mkdir(path)

arxiv = datasets.load_dataset("arxiv_dataset", data_dir=path, ignore_verifications=True)
arxiv_corpus = arxiv['train']['abstract']
arxiv_label = arxiv['train']['categories']
corpus_len = len(arxiv_corpus)  # should be 1971283

train_path = path + "/train.txt"
test_path = path + "/test.txt"
corpus_path = path + "/corpus.txt"
embedding_path = "../pretrained_embeddings/arxiv_pretrained.txt"


test_size = 443000
train_size = 664000
zipped = list(zip(arxiv_corpus, arxiv_label))
random.shuffle(zipped)
shuffled_corpus, shuffled_label = zip(*zipped)

print('start')
with open(train_path, 'w') as file:
    counter = 0
    for i in range(train_size):
        label = shuffled_label[i]
        text = shuffled_corpus[i].lower()
        text = text.replace("\n", " ")
        temp = label + "\t" + "%s\n" % text
        file.write(temp)

with open(test_path, 'w') as file:
    for i in range(train_size, train_size + test_size):
        label = shuffled_label[i]
        text = shuffled_corpus[i].lower()
        text = text.replace("\n", " ")
        temp = label + "\t" + "%s\n" % text
        file.write(temp)

# still train embeddings on the whole corpus
with open(corpus_path, 'w') as file:
    for lines in arxiv_corpus:
        file.write(lines.lower())

print('next')
model = Word2Vec(corpus_file=corpus_path, vector_size=300)
word_vector = model.wv
del model
word_vector.save_word2vec_format(embedding_path)
print('done')

