import numpy as np
import en_vectors_web_lg
import torch
import random

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from dataset_readers.data_utils import proc_seqs_pad, proc_tags_pad,\
                                       proc_seqs, proc_tags
from collections import defaultdict


class Loader(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def load(self):
        """
        Load dataset.
        """
        if self.cfgs.MODEL == 'two-pass':
            return self._load_sequence_label_revision()
        else:
            return self._load_sequence_labelling()

    def _load_sequence_labelling(self):
        """
        Load sequence labelling dataset, return dict of train,
        valid and test data.
        """
        data_dict = {}
        for split in self.cfgs.SPLIT:
            sentence_list = []
            tag_list = []
            split_list = self.cfgs.SPLIT[split].split('+')

            for item in split_list:
                with open(self.cfgs.DATA_PATH[self.cfgs.DATASET][item], 'r') as f:
                    sentence_iter = []
                    tag_iter = []

                    for line in f:
                        # each sentence are separated by double newline
                        if line != '\n':
                            word, label = line.split()

                            sentence_iter.append(word)
                            tag_iter.append(label)

                        else:
                            if len(sentence_iter) <= self.cfgs.MAX_TOKEN:
                                sentence_list.append(tuple(sentence_iter))
                                tag_list.append(tuple(tag_iter))

                            sentence_iter = []
                            tag_iter = []

            data_dict[split] = (sentence_list, tag_list)

        return data_dict

    def _load_sequence_label_revision(self):
        """
        Load sequence labelling dataset with revision signals,
        return dict of train, valid and test data.
        """
        data_dict = {}
        for split in self.cfgs.SPLIT:
            sentence_list = []
            tag_list = []
            revision_list = []
            split_list = self.cfgs.SPLIT[split].split('+')

            for item in split_list:
                with open(self.cfgs.DATA_PATH[self.cfgs.DATASET][item], 'r') as f:
                    sentence_iter = []
                    tag_iter = []
                    revision_iter = []

                    for line in f:
                        # each sentence are separated by double newline
                        if line != '\n':
                            if split == 'train' or (split == 'valid' and self.cfgs.TRAIN_SPLIT == 'train+valid'):
                                word, label, signal = line.split()
                                revision_iter.append(int(signal))
                            else:
                                word, label = line.split()

                            sentence_iter.append(word)
                            tag_iter.append(label)

                        else:
                            if len(sentence_iter) <= self.cfgs.MAX_TOKEN:
                                sentence_list.append(tuple(sentence_iter))
                                tag_list.append(tuple(tag_iter))
                                if split == 'train':
                                    revision_list.append(tuple(revision_iter))
                                    revision_iter = []

                            sentence_iter = []
                            tag_iter = []

            if split == 'train':
                data_dict[split] = (sentence_list, tag_list, revision_list)
            else:
                data_dict[split] = (sentence_list, tag_list)

        return data_dict


class SeqTokenizer(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.token2idx = {
            'PADDING': 0,
            'UNK': 1,
            'NULL': 2,  # for delayed output
        }

        self.label2idx = {
            'PADDING': 0
        }

        self.pretrained_emb = []

        if cfgs.USE_GLOVE:
            self.spacy_tool = en_vectors_web_lg.load()
            self.pretrained_emb.append(self.spacy_tool('PADDING').vector)
            self.pretrained_emb.append(self.spacy_tool('UNK').vector)
            self.pretrained_emb.append(self.spacy_tool('NULL').vector)

    def tokenize_label(self, data):
        """
        Create tokens to index, tags to index map, and embeddings
        for sequence labelling task.
        """
        for split in data:
            if split in ['train']:
                sentence_list, tag_list = data[split]

                for sentence_iter, tag_iter in zip(sentence_list, tag_list):
                    for word, label in zip(sentence_iter, tag_iter):
                        if word not in self.token2idx:
                            self.token2idx[word] = len(self.token2idx)
                            if self.cfgs.USE_GLOVE:
                                self.pretrained_emb.append(self.spacy_tool(word).vector)

                        if label not in self.label2idx:
                            self.label2idx[label] = len(self.label2idx)

            else:
                # Only get labels unseen in the training set to avoid errors
                _, tag_list = data[split]

                for tag_iter in tag_list:
                    for label in tag_iter:
                        if label not in self.label2idx:
                            self.label2idx[label] = len(self.label2idx)

        if self.cfgs.USE_GLOVE:
            self.pretrained_emb = np.array(self.pretrained_emb)

    def tokenize_label_revision(self, data):
        """
        Create tokens to index, labels to index map, and embeddings
        for sequence labelling task with revision signals.
        """
        for split in data:
            if split in ['train']:
                sentence_list, tag_list, _ = data[split]

                for sentence_iter, tag_iter in zip(sentence_list, tag_list):
                    for word, label in zip(sentence_iter, tag_iter):
                        if word not in self.token2idx:
                            self.token2idx[word] = len(self.token2idx)
                            if self.cfgs.USE_GLOVE:
                                self.pretrained_emb.append(self.spacy_tool(word).vector)

                        if label not in self.label2idx:
                            self.label2idx[label] = len(self.label2idx)

            else:
                # Only get labels unseen in the training set to avoid errors
                _, tag_list = data[split]

                for tag_iter in tag_list:
                    for label in tag_iter:
                        if label not in self.label2idx:
                            self.label2idx[label] = len(self.label2idx)

        if self.cfgs.USE_GLOVE:
            self.pretrained_emb = np.array(self.pretrained_emb)


class SeqLabellingDataset(Dataset):
    """
    Dataset object for sequence labelling.
    """
    def __init__(self, cfgs, data, tokenizer, train=True):
        super(SeqLabellingDataset, self).__init__()
        self.cfgs = cfgs
        self.train = train
        self.sequence_list, self.tag_list = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence_iter = self.sequence_list[idx]
        tag_iter = self.tag_list[idx]

        if self.cfgs.MODEL == 'two-pass':
            sent_tensor_iter = proc_seqs(sentence_iter, self.tokenizer.token2idx,
                                     self.train, self.cfgs.UNK_PROB)

            tag_tensor_iter = proc_tags(tag_iter, self.tokenizer.label2idx)
        else:
            sent_tensor_iter = proc_seqs_pad(sentence_iter, self.tokenizer.token2idx,
                                         self.cfgs.MAX_TOKEN, self.train,
                                         self.cfgs.UNK_PROB)

            tag_tensor_iter = proc_tags_pad(tag_iter, self.tokenizer.label2idx,
                                        self.cfgs.MAX_TOKEN)

        return sent_tensor_iter, tag_tensor_iter

    def __len__(self):
        return self.sequence_list.__len__()


class SeqLabellingRevisionDataset(Dataset):
    """
    Dataset object for sequence labelling with revision signals.
    """
    def __init__(self, cfgs, data, tokenizer, train=True):
        super(SeqLabellingRevisionDataset, self).__init__()
        self.cfgs = cfgs
        self.train = train
        self.sequence_list, self.tag_list, self.revision_list = data

        assert len(self.sequence_list) == len(self.tag_list) == len(self.revision_list), \
        "Mismatch of either sequence, tag or revision list."

        self.tokenizer = tokenizer
        self.sents_length = [len(i) for i in self.sequence_list]

    def __getitem__(self, idx):
        sentence_iter = self.sequence_list[idx]
        tag_iter = self.tag_list[idx]
        revision_iter = self.revision_list[idx]

        sent_tensor_iter = proc_seqs(sentence_iter, self.tokenizer.token2idx,
                                     self.train, self.cfgs.UNK_PROB)

        tag_tensor_iter = proc_tags(tag_iter, self.tokenizer.label2idx)

        # https://github.com/pytorch/pytorch/issues/2220
        revision_tensor_iter = torch.tensor(revision_iter, dtype=torch.float)

        return sent_tensor_iter, tag_tensor_iter, revision_tensor_iter

    def __len__(self):
        return self.sequence_list.__len__()


def bucket_collate(batch):
    """
    Padding for sequences of variable size length
    """
    seq_batch, tag_batch, revision_batch = [], [], []

    for (seq, tag, revision) in batch:
        seq_batch.append(seq)
        tag_batch.append(tag)
        revision_batch.append(revision)

    return pad_sequence(seq_batch, batch_first=True), \
           pad_sequence(tag_batch, batch_first=True), \
           pad_sequence(revision_batch, batch_first=True, padding_value=-1)  # because revision use BCELoss.


class BucketSampler(Sampler):
    """
    https://github.com/pytorch/pytorch/issues/46176
    """
    def __init__(self, lengths, buckets=(0, 200, 25), shuffle=True, batch_size=32, drop_last=False):
        super(BucketSampler, self).__init__(lengths)

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        assert isinstance(buckets, tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0

        buckets = defaultdict(list)
        for i, length in enumerate(lengths):
            if length > bmin:
                bucket_size = min((length // bstep) * bstep, bmax)
                buckets[bucket_size].append(i)

        self.buckets = dict()
        for bucket_size, bucket in buckets.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket, dtype=torch.int)

        # call __iter__() to store self.length
        self.__iter__()

    def __iter__(self):

        if self.shuffle:
            for bucket_size in self.buckets.keys():
                self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]

        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket, self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)

        if self.shuffle == True:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.length


class SeqLabellingDataModule(LightningDataModule):
    """
    Data module for sequence labelling.
    """
    def __init__(self, cfgs, valid=False):
        super(SeqLabellingDataModule, self).__init__()
        self.cfgs = cfgs
        self.data_loader = Loader(cfgs)
        self.tokenizer = SeqTokenizer(cfgs)
        self.data = self.data_loader.load()
        self.valid = valid

    def prepare_data(self):
        self.tokenizer.tokenize_label(self.data)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = SeqLabellingDataset(self.cfgs, self.data['train'],
                                                 self.tokenizer, train=True)

            self.valid_set = SeqLabellingDataset(self.cfgs, self.data['valid'],
                                                 self.tokenizer, train=False)

        if stage == 'test' or stage is None:
            if self.valid:
                self.test_set = SeqLabellingDataset(self.cfgs, self.data['valid'],
                                                    self.tokenizer, train=False)
            else:
                self.test_set = SeqLabellingDataset(self.cfgs, self.data['test'],
                                                    self.tokenizer, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfgs.BATCH_SIZE,
                          shuffle=True, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=1,
                          shuffle=False, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1,
                          shuffle=False, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def embedding(self):
        """
        Return GloVe embeddings
        """
        return self.tokenizer.pretrained_emb


class SeqLabellingRevisionDataModule(LightningDataModule):
    """
    Data module for sequence labelling with revision signals.
    """
    def __init__(self, cfgs, valid=False):
        super(SeqLabellingRevisionDataModule, self).__init__()
        self.cfgs = cfgs
        self.data_loader = Loader(cfgs)
        self.tokenizer = SeqTokenizer(cfgs)
        self.data = self.data_loader.load()
        self.valid = valid

    def prepare_data(self):
        self.tokenizer.tokenize_label_revision(self.data)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = SeqLabellingRevisionDataset(self.cfgs, self.data['train'],
                                                         self.tokenizer, train=True)
            self.sampler = BucketSampler(self.train_set.sents_length, buckets=(0, self.cfgs.MAX_TOKEN, 25),
                                         batch_size=self.cfgs.BATCH_SIZE)

            self.valid_set = SeqLabellingDataset(self.cfgs, self.data['valid'],
                                                 self.tokenizer, train=False)

        if stage == 'test' or stage is None:
            if self.valid:
                self.test_set = SeqLabellingDataset(self.cfgs, self.data['valid'],
                                                    self.tokenizer, train=False)
            else:
                self.test_set = SeqLabellingDataset(self.cfgs, self.data['test'],
                                                    self.tokenizer, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM, collate_fn=bucket_collate,
                          batch_sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=1,
                          shuffle=False, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1,
                          shuffle=False, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def embedding(self):
        """
        Return GloVe embeddings
        """
        return self.tokenizer.pretrained_emb
