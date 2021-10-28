import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, embed_dim, class_size, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.class_size = class_size
        self.dropout = nn.Dropout(dropout)
        self.embedder = nn.Embedding(input_size, class_size)
        self.BiLSTM = nn.LSTM(input_size, hidden_dim, bidirectional=True)
        self.linear_layer = nn.Linear(2*hidden_dim, class_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, state):
        # x: seq_len x batch_size x input_size
        x_embed = self.embedder(x) # seq_len x batch_size x class_size
        x_embed = self.dropout(x_embed)
        lstm_out, (h_n, c_n) = self.BiLSTM(x_embed, state) # lstm_out: seq_len x batch_size x 2*hidden_dim; h_n, c_n: 2 x batch_size x hidden_dim
        lstm_out = self.dropout(lstm_out)
        x_max = torch.max(lstm_out, dim=-1) # seq_len x batch_size x 2*hidden_dim
        logits = self.linear_layer(x_max) # seq_len x batch_size x class_size
        probs = self.softmax(logits) # seq_len x batch_size x class_size

        return probs, (h_n, c_n)


