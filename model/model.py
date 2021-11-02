import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.init as weight_init


class CustomEmbedding(nn.Module):
    def __init__(self, args):
        super(CustomEmbedding, self).__init__()
        self.args = args
        pretrained_embeddings = np.load(os.path.join(args.data_folder, args.dataset_name + '.word_vectors.npy'))
        pretrained_embeddings = torch.from_numpy(pretrained_embeddings).type(torch.FloatTensor)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.word_dropout = nn.Dropout(p=args.embedding_dropout)

    def forward(self, indices):
        embedded = self.embedding(indices)
        dropped = self.word_dropout(embedded)
        # dropped = dropped.type(torch.cuda.FloatTensor)
        # print(f'curr {embedded.get_device()}')
        return dropped


class CustomLSTM(nn.Module):
    def __init__(self, args):
        super(CustomLSTM, self).__init__()
        self.input_dropout = nn.Dropout(p=args.LSTM_input_dropout)
        self.LSTM = nn.LSTM(input_size=args.LSTM_input_dim,
                            hidden_size=args.LSTM_hidden_dim,
                            num_layers=args.LSTM_num_layers,
                            dropout=args.LSTM_internal_dropout,
                            bidirectional=True)
        self.output_dropout = nn.Dropout(args.LSTM_output_dropout)

    def forward(self, embedded, input_dict):
        sent_lens = input_dict['seq_length_list']
        sent_lens = torch.tensor(sent_lens).type(torch.LongTensor)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, sent_lens, enforce_sorted=False)
        output_packed, state = self.LSTM(packed_input)
        output, sent_len_output = nn.utils.rnn.pad_packed_sequence(output_packed)
        output = output.transpose(0, 1)
        return output, state


class CustomClassifier(nn.Module):
    def __init__(self, args):
        super(CustomClassifier, self).__init__()
        self.linear = nn.Linear(2 * args.LSTM_hidden_dim, args.num_labels)
        nn.init.xavier_uniform_(self.linear.weight.data)

    def forward(self, LSTM_output):
        LSTM_output = LSTM_output.transpose(1, 2)
        pooled, _ = torch.max(LSTM_output, 2)
        output = self.linear(pooled)
        return output




