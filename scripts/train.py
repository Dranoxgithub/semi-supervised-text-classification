import torch
import torch.nn as nn
from model.model import CustomEmbedding, CustomLSTM, CustomClassifier
from model.loss import at_loss, vat_loss, EM_loss
import torch.optim
import torch.nn.functional as F
import progressbar
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os


class Trainer:
    def __init__(self, data_loaders, dataset_len_dict, device, args):
        self.args = args
        self.train_loader = data_loaders['train']
        self.unlabeled_loader = data_loaders['unlabel']
        self.valid_loader = data_loaders['valid']
        self.custom_embedding = CustomEmbedding(args).to(device)
        self.custom_LSTM = CustomLSTM(args).to(device)
        self.custom_classifier = CustomClassifier(args).to(device)
        self.dataset_len_dict = dataset_len_dict
        self.device = device
        if args.enable_logging is True:
            comment = f' dataset={args.dataset_name} epochs={args.num_epochs} CE={args.use_CE} AT={args.use_AT}' \
                      f' VAT={args.use_VAT} EM={args.use_EM}'
            self.writer = SummaryWriter(comment=comment)

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

        unlabeled_iterator = iter(self.unlabeled_loader)

        self.evaluate(self.valid_loader, self.dataset_len_dict['valid'], "valid")

        for epoch in range(self.args.num_epochs):
            self.custom_embedding.train()
            self.custom_LSTM.train()
            self.custom_classifier.train()
            print('Epoch ', epoch)
            total_loss = 0
            num_processed = 0
            bar = progressbar.ProgressBar(max_value=self.dataset_len_dict['train'], redirect_stdout=True)
            for i, input_dict in enumerate(self.train_loader):
                try:
                    unlabeled_input_dict = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(self.unlabeled_loader)
                    unlabeled_input_dict = next(unlabeled_iterator)

                try:
                    batch_size = input_dict['labels'].shape[0]
                except TypeError:
                    print(input_dict)
                    raise TypeError

                # batch_dict.keys() ['batch_size', 'text', 'labels', 'seq_length_list']
                X, Y = input_dict['text'], input_dict['labels']  # X = sent_le*bsz, Y = bsz
                X_unlabeled = unlabeled_input_dict['text']

                X, Y = X.to(self.device), Y.to(self.device)
                X_unlabeled = X_unlabeled.to(self.device)

                embedded = self.custom_embedding(X)  # sent_len*bsz*embedding_dim
                unlabeled_embedded = self.custom_embedding(X_unlabeled)

                lstm_out, state = self.custom_LSTM(embedded, input_dict)  # lstm_out = bsz*sent_len*(hidden_dim*2)
                unlabeled_lstm_out, unlabeled_state = self.custom_LSTM(unlabeled_embedded, unlabeled_input_dict)

                logits = self.custom_classifier(lstm_out)  # bsz * num_classes
                unlabeled_logits = self.custom_classifier(unlabeled_lstm_out)

                normalized_probs = F.log_softmax(logits, dim=-1)
                loss = 0
                # CE loss
                if self.args.use_CE:
                    CE_loss = self.args.ml_loss_weight * criterion(normalized_probs, Y)
                    loss += CE_loss

                # at loss
                if self.args.use_AT:
                    AT_loss = self.args.at_loss_weight * at_loss(input_dict, self.custom_embedding, self.custom_LSTM,
                                                                 self.custom_classifier,
                                                                 X, Y, at_epsilon=self.args.at_epsilon)
                    loss += AT_loss

                # vat loss
                if self.args.use_VAT:
                    VAT_loss = self.args.vat_loss_weight * vat_loss(self.device, input_dict, self.custom_embedding,
                                                                    self.custom_LSTM, self.custom_classifier,
                                                                    X, logits.detach(), self.args.vat_epsilon,
                                                                    self.args.hyperpara_for_vat)
                    VAT_loss_unlabel = self.args.vat_loss_weight * vat_loss(self.device, unlabeled_input_dict, self.custom_embedding,
                                                                    self.custom_LSTM, self.custom_classifier,
                                                                    X, unlabeled_logits.detach(), self.args.vat_epsilon,
                                                                    self.args.hyperpara_for_vat)
                    loss += VAT_loss
                    loss += VAT_loss_unlabel

                # EM loss
                if self.args.use_EM:
                    labeled_entropy = EM_loss(logits)
                    unlabeled_entropy = EM_loss(unlabeled_logits)
                    averaged_entropy = 0.5 * (labeled_entropy + unlabeled_entropy)
                    EM_loss_val = self.args.EM_loss_weight * averaged_entropy
                    loss += EM_loss_val

                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],
                                               self.args.gradient_clip_value)
                optimizer.step()
                num_processed += batch_size
                bar.update(num_processed)

                num_correct = self.get_num_correct(normalized_probs.cpu().detach(), Y.cpu().detach())
                acc = num_correct / batch_size

                if i % self.args.eval_freq == 0:
                    print(f'batch {i:04} accuracy: {acc:.2f}')

                if i % self.args.logging_freq == 0:
                    if self.args.enable_logging is True:
                        self.writer.add_scalar("training accuracy", acc, i)
                        self.writer.add_scalar("training total loss", loss.detach().cpu().numpy(), i)
                        if self.args.use_CE is True:
                            self.writer.add_scalar("CE loss", CE_loss.detach().cpu().numpy(), i)
                        if self.args.use_AT is True:
                            self.writer.add_scalar("AT loss", AT_loss.detach().cpu().numpy(), i)
                        if self.args.use_VAT is True:
                            self.writer.add_scalar("VAT loss", VAT_loss.detach().cpu().numpy(), i)
                        if self.args.use_EM is True:
                            self.writer.add_scalar("EM loss", EM_loss_val.detach().cpu().numpy(), i)

            eval_acc = self.evaluate(self.valid_loader, self.dataset_len_dict['valid'], "valid")
            if self.args.enable_logging is True:
                self.writer.add_scalar("validation accuracy", eval_acc, epoch)
            print(f'Loss for epoch {epoch} : {total_loss}')
        if self.args.enable_logging is True:
            self.writer.flush()
            self.writer.close()


    def evaluate(self, dataloader, data_length, dataset_type):
        print(f"Entering evaluation on {dataset_type}")
        self.custom_embedding.eval()
        self.custom_LSTM.eval()
        self.custom_classifier.eval()

        total_num_correct = 0
        num_processed = 0
        bar = progressbar.ProgressBar(max_value=data_length, redirect_stdout=True)
        for i, input_dict in enumerate(dataloader):
            try:
                batch_size = input_dict['labels'].shape[0]
            except TypeError:
                print(input_dict)
                raise TypeError

            # batch_dict.keys() ['batch_size', 'text', 'labels', 'seq_length_list']
            X, Y = input_dict['text'], input_dict['labels']
            X, Y = X.to(self.device), Y.to(self.device)  # X = padded_sent_le*batch size
            embedded = self.custom_embedding(X)

            lstm_out, state = self.custom_LSTM(embedded, input_dict)
            clf_out = self.custom_classifier(lstm_out)
            logits = F.log_softmax(clf_out, dim=-1)

            num_processed += batch_size
            bar.update(num_processed)

            num_correct = self.get_num_correct(logits.cpu().detach(), Y.cpu().detach())
            total_num_correct += num_correct
        acc = total_num_correct / data_length
        print(f'Accuracy on {dataset_type} dataset: {acc:.2f}')
        return acc
