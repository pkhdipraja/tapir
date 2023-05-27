import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "Create a Layernorm module."
    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    "Positional encoding function."
    def __init__(self, cfgs, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.cfgs = cfgs
        self.dropout = nn.Dropout(cfgs.DROPOUT)

        pe = torch.zeros(max_len, cfgs.HIDDEN_SIZE)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, cfgs.HIDDEN_SIZE, 2) *
                             - (math.log(10000.0) / cfgs.HIDDEN_SIZE))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x)


def make_mask(feature):
    mask = (torch.sum(
        torch.abs(feature), dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)
    return mask


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool),
                                 diagonal=1)
    return subsequent_mask


def add_null_tokens(x, delay, null_token):
    "Add null tokens as in Turek et al. (2020)."
    delay_mask = torch.zeros(x.shape, dtype=torch.bool)
    seq_len = torch.sum(torch.abs(x) != 0, dim=-1).view(-1, 1)
    row_idx = torch.arange(x.size(0), dtype=torch.long).view(-1, 1)
    null_idx = []
    for i in range(delay):
        null_idx.append(seq_len)
        seq_len = seq_len + 1

    null_idx = torch.cat(null_idx, dim=1)
    delay_mask[row_idx, null_idx] = True
    delay_mask = delay_mask.to(x.device)
    x = x.masked_fill(delay_mask, null_token)
    return x


def rnn_add_null_tokens(x, delay, null_token):
    "Add null tokens as in Turek et al. (2020). for RNN datasets"
    assert delay > 0, "Invalid delay value."
    delay_x = F.pad(x, (0,delay), "constant", 0)
    delay_mask = torch.zeros(delay_x.shape, dtype=torch.bool)

    seq_len = torch.sum(torch.abs(delay_x) != 0, dim=-1).view(-1, 1)
    row_idx = torch.arange(delay_x.size(0), dtype=torch.long).view(-1, 1)
    null_idx = []
    for i in range(delay):
        null_idx.append(seq_len)
        seq_len = seq_len + 1

    null_idx = torch.cat(null_idx, dim=1)
    delay_mask[row_idx, null_idx] = True
    delay_mask = delay_mask.to(x.device)
    delay_x = delay_x.masked_fill(delay_mask, null_token)
    return delay_x
