import torch
import torch.nn as nn
import torch.nn.functional as F

def at_loss(input_dict, custom_embedding, custom_LSTM, custom_classifier, X, Y, at_epsilon=5.0):
    criterion = nn.NLLLoss()
    embedded = custom_embedding(X) # sent_len*bsz*embedding_dim
    embedded.retain_grad()
    lstm_out, state = custom_LSTM(embedded, input_dict)
    clf_out = custom_classifier(lstm_out)
    logits = F.log_softmax(clf_out, dim=-1)

    loss = criterion(logits, Y)
    loss.backward()

    g = normalize_matrix(embedded.grad.data)
    pert_embedded = embedded + at_epsilon * g
    lstm_out, state = custom_LSTM(pert_embedded, input_dict)
    clf_out = custom_classifier(lstm_out)
    logits = F.log_softmax(clf_out, dim=-1)

    return criterion(logits, Y)
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