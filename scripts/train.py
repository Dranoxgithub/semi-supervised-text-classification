import torch
from tqdm import tqdm
import torch.nn as nn
from model.model import BiLSTMModel
import torch.optim


class BiLSTM:
    def __init__(self, hidden_dim, class_size, embeddings, dropout=0.5):
        self.model = BiLSTMModel(hidden_dim, class_size, embeddings, dropout)

    def train(self, train_loader):
        loss_fn = nn.CrossEntropyLoss
        n_epoch = 20 # 20 for supervised training, 50 for unsupervised
        lr = 1e-3
        b1 = 0
        b2 = 0.98
        optimizer = torch.optim.Adam(self.model.parameters(), lr, betas=(b1, b2))
        for epoch in range(n_epoch):
            print('Epoch ', epoch)
            loss_per_epoch = 0
            for batch in tqdm(train_loader):
                X, Y = batch
                pred_probs = self.model(X)
                loss = loss_fn(pred_probs, Y)
                loss_per_epoch += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Loss for epoch {epoch} : {loss_per_epoch}')