import torch
import torch.nn as nn
from model.model import CustomEmbedding, CustomLSTM, CustomClassifier
from model.loss import at_loss, vat_loss, EM_loss
import torch.optim
import torch.nn.functional as F
import progressbar
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
            comment = f' dset={args.dataset_name} CE({args.ml_loss_weight}x)={args.use_CE} AT({args.at_loss_weight}x)={args.use_AT}' \
                      f' VAT({args.vat_loss_weight}x)={args.use_VAT} EM({args.em_loss_weight}x)={args.use_EM}'
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

    def log_gradients_model(self, step):
        for tag, value in self.custom_embedding.named_parameters():
            if value.grad is not None:
                self.writer.add_histogram(tag + "/grad", value.grad.cpu(), step)
        for tag, value in self.custom_LSTM.named_parameters():
            if value.grad is not None and tag == 'LSTM.weight_hh_l0':
                self.writer.add_histogram(tag + "/grad", value.grad.cpu(), step)
        for tag, value in self.custom_classifier.named_parameters():
            if value.grad is not None:
                self.writer.add_histogram(tag + "/grad", value.grad.cpu(), step)

    def train(self):
        batch_number = 0
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.collect_trainable_params(), lr=self.args.lr, betas=self.args.betas)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998)
        unlabeled_iterator = iter(self.unlabeled_loader)

        self.evaluate(self.valid_loader, self.dataset_len_dict['valid'], "valid")

        for epoch in range(self.args.num_epochs):
            self.custom_embedding.train()
            self.custom_LSTM.train()
            self.custom_classifier.train()
            print('Epoch ', epoch)
            total_loss = 0
            total_CE_loss = 0
            total_EM_loss = 0
            total_AT_loss = 0
            total_VAT_loss = 0
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
                    total_CE_loss += CE_loss

                # at loss
                if self.args.use_AT:
                    AT_loss = self.args.at_loss_weight * at_loss(input_dict, self.custom_embedding, self.custom_LSTM,
                                                                 self.custom_classifier,
                                                                 X, Y, at_epsilon=self.args.at_epsilon)
                    loss += AT_loss
                    total_AT_loss += AT_loss

                # vat loss
                if self.args.use_VAT:
                    VAT_loss = self.args.vat_loss_weight * vat_loss(self.device, input_dict, self.custom_embedding,
                                                                    self.custom_LSTM, self.custom_classifier,
                                                                    X, logits.detach(), self.args.vat_epsilon,
                                                                    self.args.hyperpara_for_vat)

                    VAT_loss_unlabel = self.args.vat_loss_weight * vat_loss(self.device, unlabeled_input_dict,
                                                                            self.custom_embedding,
                                                                            self.custom_LSTM, self.custom_classifier,
                                                                            X_unlabeled, unlabeled_logits.detach(),
                                                                            self.args.vat_epsilon,
                                                                            self.args.hyperpara_for_vat)
                    ratio = len(X) / (len(X) + len(X_unlabeled))
                    averaged_VAT_loss = ratio * VAT_loss + (1 - ratio) * VAT_loss_unlabel
                    loss = loss + averaged_VAT_loss
                    total_VAT_loss += averaged_VAT_loss

                # EM loss
                if self.args.use_EM:
                    combined_len = logits.size(dim=0) + unlabeled_logits.size(dim=0)
                    labeled_entropy = EM_loss(logits)
                    unlabeled_entropy = EM_loss(unlabeled_logits)

                    averaged_entropy = (logits.size(dim=0)/combined_len) * labeled_entropy + \
                                       (unlabeled_logits.size(dim=0)/combined_len) * unlabeled_entropy
                    EM_loss_val = self.args.em_loss_weight * averaged_entropy
                    loss += EM_loss_val
                    total_EM_loss += EM_loss_val

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
                        self.writer.add_scalar("training accuracy", acc, batch_number)
                        self.log_gradients_model(batch_number)

                batch_number += 1

            eval_acc = self.evaluate(self.valid_loader, self.dataset_len_dict['valid'], "valid")

            if self.args.enable_logging is True:
                self.writer.add_scalar("training total loss", total_loss.detach().cpu().numpy(), epoch)
                self.writer.add_scalar("validation accuracy", eval_acc, epoch)
                if self.args.use_CE is True:
                    self.writer.add_scalar(f"CE loss", total_CE_loss.detach().cpu().numpy(), epoch)
                if self.args.use_AT is True:
                    self.writer.add_scalar(f"AT loss", total_AT_loss.detach().cpu().numpy(), epoch)
                if self.args.use_VAT is True:
                    self.writer.add_scalar(f"VAT loss", total_VAT_loss.detach().cpu().numpy(), epoch)
                if self.args.use_EM is True:
                    self.writer.add_scalar(f"EM loss", total_EM_loss.detach().cpu().numpy(), epoch)

            print(f'Loss for epoch {epoch} : {total_loss}')
            scheduler.step()

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
