import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import PositionalEncoding
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask


class LinearCausalEncoderLabelling(nn.Module):
    """
    N stack of Linear Transformers with causal masking for
    sequence labelling.
    """
    def __init__(self, cfgs, token_size,
                 pretrained_emb=None, position_enc=True):
        super(LinearCausalEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.params = {
            'attention_type': 'causal-linear',
            'n_layers': cfgs.LAYER,
            'n_heads': cfgs.ATTENTION_HEAD,
            'feed_forward_dimensions': cfgs.FF_SIZE,
            'query_dimensions': self.HIDDEN_SIZE_HEAD,
            'value_dimensions': self.HIDDEN_SIZE_HEAD,
            'dropout': cfgs.DROPOUT,
            'attention_dropout': cfgs.DROPOUT,
            'activation': 'relu',
            'final_normalization': True
        }

        self.encoder = TransformerEncoderBuilder.from_dictionary(self.params).get()
        self.position_enc = position_enc

        if cfgs.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.proj = nn.Linear(cfgs.WORD_EMBED_SIZE,
                                  cfgs.HIDDEN_SIZE)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.HIDDEN_SIZE
            )

        if self.position_enc:
            self.position = PositionalEncoding(cfgs)

        # Xavier init
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        length_mask = LengthMask(
            torch.sum(torch.abs(x) != 0, dim=-1),
            max_len=self.cfgs.MAX_TOKEN
        )

        attn_mask = TriangularCausalMask(self.cfgs.MAX_TOKEN)

        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)

        x = self.encoder(
            x, attn_mask=attn_mask,
            length_mask=length_mask
        )

        return x


class LinearEncoderLabelling(nn.Module):
    """
    N stack of Linear Transformers for sequence labelling.
    """
    def __init__(self, cfgs, token_size,
                 pretrained_emb=None, position_enc=True):
        super(LinearEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.params = {
            'attention_type': 'linear',
            'n_layers': cfgs.LAYER,
            'n_heads': cfgs.ATTENTION_HEAD,
            'feed_forward_dimensions': cfgs.FF_SIZE,
            'query_dimensions': self.HIDDEN_SIZE_HEAD,
            'value_dimensions': self.HIDDEN_SIZE_HEAD,
            'dropout': cfgs.DROPOUT,
            'attention_dropout': cfgs.DROPOUT,
            'activation': 'relu',
            'final_normalization': True
        }

        self.encoder = TransformerEncoderBuilder.from_dictionary(self.params).get()
        self.position_enc = position_enc

        if cfgs.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.proj = nn.Linear(cfgs.WORD_EMBED_SIZE,
                                  cfgs.HIDDEN_SIZE)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.HIDDEN_SIZE
            )

        if self.position_enc:
            self.position = PositionalEncoding(cfgs)

        # Xavier init
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        length_mask = LengthMask(
            torch.sum(torch.abs(x) != 0, dim=-1),
            max_len=x.size(-1)
        )

        attn_mask = None  # automatically attend to all tokens

        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)

        x = self.encoder(
            x, attn_mask=attn_mask,
            length_mask=length_mask
        )

        return x
