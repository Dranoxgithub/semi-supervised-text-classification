import pickle
import torch
from torchtext import data
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler

class customDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][1], self.dataset[idx][0]

class CustomLoader(object):
    def __init__(self, dataset, num_token_per_batch, collate_fn=None):
        self.dataset = dataset
        self.num_token_per_batch = num_token_per_batch
        self.sampler = RandomSampler(dataset)
        self.collate_fn = collate_fn

    def __iter__(self):
        batch = []
        curr_length = 0
        for idx in self.sampler:
            batch.append(self.dataset[idx])
            curr_length += len(self.dataset[idx][0])
            if curr_length >= self.num_token_per_batch: 
                if self.collate_fn is not None:
                    collated_batch = self.collate_fn(batch)
                    yield collated_batch
                batch = []
                curr_length = 0
        if self.collate_fn is not None:
            collated_batch = self.collate_fn(batch)
            yield collated_batch


def load(train_set, valid_set, test_set, unlabel_set, num_token_per_batch):
    dset = pickle.load(open("/usr/xtmp/ac638/others/semi-supervised-text-classification/temp/aclImdbSimple_pretrained_mixed/aclImdb_tok.train.pkl", 'rb'))
    # dset = [(0,[1]), (1, [2,2]), (3, [4,4,4,4]), (3, [4,4,4,4, 6]), (3, [4,4,4,4,7])]
    # print(dset[0])

    dl = CustomLoader(customDataset(dset), num_token_per_batch=1000, collate_fn=collate_fn)

    train = pickle.load(open(train_set, 'rb'))
    valid = pickle.load(open(valid_set, 'rb'))
    test = pickle.load(open(test_set, 'rb'))
    unlabel = pickle.load(open(unlabel_set, 'rb'))

    train_dataset = CustomLoader(customDataset(train), num_token_per_batch=num_token_per_batch, collate_fn=collate_fn)
    valid_dataset = CustomLoader(customDataset(valid), num_token_per_batch=num_token_per_batch, collate_fn=collate_fn)
    test_dataset = CustomLoader(customDataset(test), num_token_per_batch=num_token_per_batch, collate_fn=collate_fn)
    unlabel_dataset = CustomLoader(customDataset(unlabel), num_token_per_batch=num_token_per_batch, collate_fn=collate_fn)

    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset, "unlabel": unlabel_dataset}


global max_len_curr_batch
def customized_batch_size_fn(incoming, count, sofar):
    global max_len_curr_batch
    if count == 1:
        max_len_curr_batch = len(incoming[1]) + 2
    else: 
        max_len_curr_batch = max(max_len_curr_batch, len(incoming[1]) + 2) # add the bos and eos tokens 
    return count * max_len_curr_batch
def iterate_data(dset, num_words_per_batch, random_shuffler=data.iterator.RandomShuffler()):
    # data: Y, X
    return data.iterator.pool(dset, num_words_per_batch, key=lambda x: len(x[1]),
    batch_size_fn=customized_batch_size_fn, random_shuffler=random_shuffler)

def collate_fn(batch):
    # pad seq length * batch_size
    word_ids, labels = zip(*batch)
    labels = torch.tensor([*labels])
    seq_length = [len(i) for i in word_ids]
    max_length = max(seq_length)
    word_list = []
    for i in word_ids:
        word_list.append(i + (max_length - len(i)) * [0])
    word_list = torch.tensor(word_list) # word_list  batch_size * seq_length
    
    return {'batch_size': word_list.shape[0], 'text': word_list, 'labels': labels, 'seq_length_list': seq_length}
    
    