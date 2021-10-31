import torch
from tqdm import tqdm
import torch.nn as nn
from model.model import BiLSTMModel
import torch.optim

class BiLSTM:
    def __init__(self, hidden_dim, class_size, embeddings, num_layers=1, dropout=0.5):
        self.model = BiLSTMModel(hidden_dim, class_size, embeddings, dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def train(self, train_loader, n_epoch, lr, b1, b2):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr, betas=(b1, b2))

        for epoch in range(n_epoch):
            print('Epoch ', epoch)
            total_loss = 0
            for index, batch_dict in enumerate(tqdm(train_loader)):
                # batch_dict.keys() ['batch_size', 'text', 'labels', 'seq_length_list']
                batch_size, X, Y, seq_length_list = batch_dict['batch_size'], batch_dict['text'], batch_dict['labels'], batch_dict['seq_length_list']
                print(X)
                print(Y)
                hidden = (torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim), torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim))
                pred_probs = self.model(X, hidden)
                loss = loss_fn(pred_probs, Y)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
            break
            print(f'Loss for epoch {epoch} : {loss_per_epoch}')