import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from model import BiLSTM
import torch.optim

class BiLSTMModel:
    def __init__(self):
        self.model = BiLSTM

    def train(self, dataset):
        train_loader = #left to be filled when dataloader is done
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