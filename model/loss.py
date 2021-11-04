import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def at_loss(input_dict, custom_embedding, custom_LSTM, custom_classifier, X, Y, at_epsilon=5.0):
    criterion = nn.NLLLoss()

    embedded = custom_embedding(X).detach().requires_grad_(True) # sent_len * bsz * embedding_dim
    lstm_out, state = custom_LSTM(embedded, input_dict)
    logit = custom_classifier(lstm_out)
    log_softmax = F.log_softmax(logit, dim=-1)
    loss = criterion(log_softmax, Y)
    loss.backward()

    g = normalize_matrix(embedded.grad.detach())
    pert_embedded = embedded + at_epsilon * g
    lstm_out, state = custom_LSTM(pert_embedded, input_dict)
    logit = custom_classifier(lstm_out)
    log_softmax = F.log_softmax(logit, dim=-1)

    return criterion(log_softmax, Y)
def normalize_matrix(matrix):
    # sent_len * bsz * embedding_dim
    matrix = torch.permute(matrix, [1, 0, 2])  # bsz * sent_len * embedding_dim
    _, sent_len, _ = matrix.shape
    matrix = matrix.reshape(matrix.shape[0], -1) # bsz * (sent_len * embedding_dim)
    # to prevent underflow when squaring, normalize first by maximum of the absolute value 
    abs_matrix = torch.abs(matrix)
    max_value_vector, _ = torch.max(abs_matrix, dim=1, keepdim=True) 
    matrix = matrix / (1e-20 + max_value_vector) 

    matrix = matrix / (torch.sqrt(torch.sum(matrix**2, dim=1, keepdim=True)) + 1e-20)
    matrix = matrix.reshape(matrix.shape[0], sent_len, -1) # bsz * sent_len * embedding_dim
    matrix = torch.permute(matrix, [1, 0, 2]) #  sent_len * bsz * embedding_dim
    return matrix

def vat_loss(device, input_dict, custom_embedding, custom_LSTM, custom_classifier, X, logit_for_v, vat_epsilon, hyperpara_for_vat):
    embedded = custom_embedding(X) # sent_len * bsz * embedding_dim
    d = torch.normal(0, 1, size=embedded.shape).to(device)

    # v_prime leaf variable
    v_prime = (embedded.detach() + hyperpara_for_vat * d).requires_grad_(True)
    lstm_out, state = custom_LSTM(v_prime, input_dict)
    logit = custom_classifier(lstm_out)
    kl_loss = kl_divergence(logit_for_v, logit)
    kl_loss.backward()
    g = normalize_matrix(v_prime.grad.detach())

    pert_embedded = embedded + vat_epsilon * g
    lstm_out, state = custom_LSTM(pert_embedded, input_dict)
    logit = custom_classifier(lstm_out)
    return kl_divergence(logit_for_v, logit)

def kl_divergence(logit_for_v, new_logit):
    assert len(logit_for_v.shape) == 2
    kl = torch.sum((F.log_softmax(logit_for_v, dim=-1) - F.log_softmax(new_logit, dim=-1)) * F.softmax(logit_for_v, dim=-1), -1)
    # average across batches
    return torch.mean(kl) 