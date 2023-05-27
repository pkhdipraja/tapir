import torch
import numpy as np


def proc_seqs_pad(seqs, token2idx, max_token, train=True, unk_prob=0.0):
    seq_tensor = torch.zeros(max_token, dtype=torch.long)

    for idx, word in enumerate(seqs):
        if word in token2idx:
            if train and np.random.uniform(0, 1) > 1 - unk_prob:
                seq_tensor[idx] = token2idx['UNK']
            else:
                seq_tensor[idx] = token2idx[word]
        else:
            seq_tensor[idx] = token2idx['UNK']

        if idx + 1 == max_token:
            break

    return seq_tensor


def proc_tags_pad(tags, label2idx, max_token):
    tag_tensor = torch.zeros(max_token, dtype=torch.long)

    for idx, tag in enumerate(tags):
        # All possible labels are already mapped beforehand
        tag_tensor[idx] = label2idx[tag]

        if idx+1 == max_token:
            break

    return tag_tensor


def proc_seqs(seqs, token2idx, train=True, unk_prob=0.0):
    seq_tensor = torch.zeros(len(seqs), dtype=torch.long)

    for idx, word in enumerate(seqs):
        if word in token2idx:
            if train and np.random.uniform(0, 1) > 1 - unk_prob:
                seq_tensor[idx] = token2idx['UNK']
            else:
                seq_tensor[idx] = token2idx[word]
        else:
            seq_tensor[idx] = token2idx['UNK']

    return seq_tensor


def proc_tags(tags, label2idx):
    tag_tensor = torch.zeros(len(tags), dtype=torch.long)

    for idx, tag in enumerate(tags):
        # All possible labels are already mapped beforehand
        tag_tensor[idx] = label2idx[tag]

    return tag_tensor


# def proc_labels(label, label2idx):
#     return torch.tensor([label2idx[label]], dtype=torch.long)
