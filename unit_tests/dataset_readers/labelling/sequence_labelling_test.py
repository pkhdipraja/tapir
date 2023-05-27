import pytest
import torch

from unit_tests.dataset_readers.labelling.conftest import DummyConfig, dummy_tokenizer, expected_data
from dataset_readers.datasets import SeqLabellingDataset, Loader, SeqTokenizer


@pytest.mark.parametrize('dataset',
('snips-slot', 'multilingual', 'movie', 'chunk-conll2003',
'pos-conll2003', 'ner-conll2003', 'pos-ud-ewt'))
def test_loader(dataset, expected_data):
    cfgs = DummyConfig()
    cfgs.DATASET = dataset

    expected_sents = expected_data[dataset]['sentences']
    expected_tags = expected_data[dataset]['tags']

    data_loader = Loader(cfgs)
    data = data_loader.load()
    sentence_list, tag_list = data['train']

    for sents, gold_sents in zip(sentence_list, expected_sents):
        assert sents == gold_sents, "Sentences does not match expected values."

    for tags, gold_tags in zip(tag_list, expected_tags):
        assert tags == gold_tags, "Tags does not match expected values."


def test_tokenizer(dummy_tokenizer):
    cfgs = DummyConfig()
    dummy_data, expected_token2idx, expected_label2idx, _ = dummy_tokenizer

    tokenizer = SeqTokenizer(cfgs)
    tokenizer.tokenize_label(dummy_data)

    token2idx = tokenizer.token2idx
    label2idx = tokenizer.label2idx

    # ensure that vocab is only added from train, however ensure that
    # label from val/test is also added.
    assert token2idx == expected_token2idx, "Token to index map does not match expected values."
    assert label2idx == expected_label2idx, "Label to index map does not match expected values."


def test_dataset(dummy_tokenizer):
    cfgs = DummyConfig()
    dummy_data, _, _, dummy_data_tensor = dummy_tokenizer

    tokenizer = SeqTokenizer(cfgs)
    tokenizer.tokenize_label(dummy_data)

    train_set = SeqLabellingDataset(cfgs, dummy_data['train'], tokenizer,
                                    train=True)

    test_set = SeqLabellingDataset(cfgs, dummy_data['test'], tokenizer,
                                   train=False)

    train_sent_iter, train_tag_iter = train_set[0]
    test_sent_iter, test_tag_iter = test_set[0]

    gold_train_sent_iter, gold_train_tag_iter = dummy_data_tensor['train']
    gold_test_sent_iter, gold_test_tag_iter = dummy_data_tensor['test']

    # ensure that tensor of sequences and label containing the idx
    # is also created similar to token2idx and label2idx map
    assert torch.equal(train_sent_iter, gold_train_sent_iter), "Sentence tensor mismatch."
    assert torch.equal(train_tag_iter, gold_train_tag_iter), "Tag tensor mismatch."
    assert torch.equal(test_sent_iter, gold_test_sent_iter), "Sentence tensor mismatch."
    assert torch.equal(test_tag_iter, gold_test_tag_iter), "Tag tensor mismatch."
