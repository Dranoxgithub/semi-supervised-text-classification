import torch
from tqdm import tqdm
import torch.nn as nn
from model.model import CustomEmbedding, CustomLSTM, CustomClassifier
import torch.optim
import torch.nn.functional as F
import progressbar
import numpy as np


class Trainer:
    def __init__(self, train_loader, dataset_len_dict, device, args):
        self.args = args
        self.train_loader = train_loader
        self.custom_embedding = CustomEmbedding(args).to(device)
        self.custom_LSTM = CustomLSTM(args).to(device)
        self.custom_classifier = CustomClassifier(args).to(device)
        self.dataset_len_dict = dataset_len_dict
        self.device = device

    @staticmethod
    def get_num_correct(out, target):
        predictions = out.max(-1)[1].type(torch.LongTensor)
        num_correct = predictions.eq(target).float()
        num_correct = torch.sum(num_correct).item()
        return num_correct

    def collect_trainable_params(self):
        param_list = list(self.custom_embedding.parameters()) + \
                     list(self.custom_LSTM.parameters()) + \
                     list(self.custom_classifier.parameters())
        return param_list

    def train(self):
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.collect_trainable_params(), self.args.lr)
        for epoch in range(self.args.num_epochs):
            print('Epoch ', epoch)
            total_loss = 0
            num_processed = 0
            bar = progressbar.ProgressBar(max_value=self.dataset_len_dict['train'], redirect_stdout=True)
            for i, input_dict in enumerate(self.train_loader):
                batch_size = input_dict['labels'].shape[0]
                # batch_dict.keys() ['batch_size', 'text', 'labels', 'seq_length_list']
                X, Y = input_dict['text'], input_dict['labels']
                X, Y = X.to(self.device), Y.to(self.device)  # X = padded_sent_le*batch size

                embedded = self.custom_embedding(X)

                lstm_out, state = self.custom_LSTM(embedded, input_dict)
                clf_out = self.custom_classifier(lstm_out)
                logits = F.log_softmax(clf_out, dim=-1)

                loss = criterion(logits, Y)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],
                                               self.args.gradient_clip_value)
                optimizer.step()
                num_processed += batch_size
                bar.update(num_processed)

                if i % 100 == 0:
                    num_correct = self.get_num_correct(logits.cpu().detach(), Y.cpu().detach())
                    acc = num_correct / batch_size
                    print(f'batch {i:04} accuracy: {acc:.2f}')
            print(f'Loss for epoch {epoch} : {total_loss}')
