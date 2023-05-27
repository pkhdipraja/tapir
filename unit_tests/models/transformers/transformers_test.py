import pytest
import torch

from model.transformer import MultiHeadedAttention, EncoderLayer
from model.model_utils import make_mask


class TransformerTestConfig(object):
    def __init__(self):
        self.HIDDEN_SIZE = 512
        self.ATTENTION_HEAD = 8
        self.DROPOUT = 0.1
        self.FF_SIZE = 2048


def test_mha_outputs():
    cfgs = TransformerTestConfig()
    embeds = torch.randn([3, 5, cfgs.HIDDEN_SIZE])
    inputs = torch.randint(10, (3, 5))
    mask = make_mask(inputs.unsqueeze(2))
    model = MultiHeadedAttention(cfgs)
    outs = model(embeds, embeds, embeds, mask)
    assert outs.shape == (3, 5, cfgs.HIDDEN_SIZE), "Multi-headed attention output does not match."


def test_encoder_outputs():
    cfgs = TransformerTestConfig()
    embeds = torch.randn([3, 5, cfgs.HIDDEN_SIZE])
    inputs = torch.randint(10, (3, 5))
    mask = make_mask(inputs.unsqueeze(2))
    model = EncoderLayer(cfgs)
    outs = model(embeds, mask)
    assert outs.shape == (3, 5, cfgs.HIDDEN_SIZE), "Encoder output shape does not match."
