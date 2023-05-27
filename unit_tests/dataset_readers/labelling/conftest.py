import torch
import pytest
import os


class DummyConfig(object):
    def __init__(self):
        self.DATASET = None
        self.TASK_TYPE = 'labelling'
        self.MODEL = None

        self.SPLIT = {
            'train': 'train'
        }

        self.PATH_PREFIX_TAGS = 'unit_tests/dataset_readers/sample_test/labelling'

        self.DATA_PATH = {
            'snips-slot': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.snips_slot')
            },
            'multilingual': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.multilingual')
            },
            'movie': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.movie')
            },
            'chunk-conll2003': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.chunk_conll2003')
            },
            'pos-conll2003': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.pos_conll2003')
            },
            'ner-conll2003': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.ner_conll2003')
            },
            'pos-ud-ewt': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.pos_ud_ewt')
            }
        }

        self.MAX_TOKEN = 60
        self.USE_GLOVE = True
        self.UNK_PROB = 0.0


@pytest.fixture
def expected_data():
    expected_dict = {
        'snips-slot': {
            'sentences': [
                ('add', 'sabrina', 'salerno', 'to', 'the', 'grime', 'instrumentals', 'playlist'),
                ('i', 'want', 'to', 'bring', 'four', 'people', 'to', 'a', 'place', 'that', 's', 
                 'close', 'to', 'downtown', 'that', 'serves', 'churrascaria', 'cuisine')
            ],
            'tags': [
                ('O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O'),
                ('O', 'O', 'O', 'O', 'B-party_size_number', 'O', 'O', 'O', 'O', 'O',
                 'O', 'B-spatial_relation', 'O', 'B-poi', 'O', 'O', 'B-restaurant_type', 'O')
            ]
        },
        'multilingual': {
            'sentences': [
                ('i', 'want', 'to', 'be', 'reminded', 'to', 'file', 'tax'),
                ('i', 'don\'t', 'want', 'to', 'forget', 'to', 'file', 'taxes')
            ],
            'tags': [
                ('O', 'O', 'O', 'O', 'O', 'O', 'B-reminder/todo', 'I-reminder/todo'),
                ('O', 'O', 'O', 'O', 'O', 'O', 'B-reminder/todo', 'I-reminder/todo')
            ]
        },
        'movie': {
            'sentences': [
                ('are', 'there', 'any', 'good', 'romantic', 'comedies', 'out', 'right', 'now'),
                ('show', 'me', 'a', 'movie', 'about', 'cars', 'that', 'talk')
            ],
            'tags': [
                ('O', 'O', 'O', 'O', 'B-GENRE', 'I-GENRE', 'O', 'B-YEAR', 'I-YEAR'),
                ('O', 'O', 'O', 'O', 'O', 'B-PLOT', 'I-PLOT', 'I-PLOT')
            ]
        },
        'chunk-conll2003': {
            'sentences': [
                ('AL-AIN', ',', 'United', 'Arab', 'Emirates', '1996-12-06'),
                ('Japan', 'began', 'the', 'defence', 'of', 'their', 'Asian', 'Cup', 'title', 'with', 'a', 'lucky', 
                '2-1', 'win', 'against', 'Syria', 'in', 'a', 'Group', 'C', 'championship', 'match', 'on', 'Friday', '.')
            ],
            'tags': [
                ('I-NP', 'O', 'I-NP', 'I-NP', 'I-NP', 'I-NP'),
                ('I-NP', 'I-VP', 'I-NP', 'I-NP', 'I-PP', 'I-NP', 'I-NP', 'I-NP', 'I-NP', 'I-PP', 'I-NP', 'I-NP', 'I-NP',
                'I-VP', 'I-PP', 'I-NP', 'I-PP', 'I-NP', 'I-NP', 'I-NP', 'I-NP', 'I-NP', 'I-PP', 'I-NP', 'O')
            ]
        },
        'pos-conll2003': {
            'sentences': [
                ('AL-AIN', ',', 'United', 'Arab', 'Emirates', '1996-12-06'),
                ('Japan', 'began', 'the', 'defence', 'of', 'their', 'Asian', 'Cup', 'title', 'with', 'a', 'lucky', 
                '2-1', 'win', 'against', 'Syria', 'in', 'a', 'Group', 'C', 'championship', 'match', 'on', 'Friday', '.')
            ],
            'tags': [
                ('NNP', ',', 'NNP', 'NNP', 'NNPS', 'CD'),
                ('NNP', 'VBD', 'DT', 'NN', 'IN', 'PRP$', 'JJ', 'NNP', 'NN', 'IN', 'DT', 'JJ', 'CD', 'VBP', 'IN', 'NNP',
                'IN', 'DT', 'NNP', 'NNP', 'NN', 'NN', 'IN', 'NNP', '.')
            ]
        },
        'ner-conll2003': {
            'sentences': [
                ('AL-AIN', ',', 'United', 'Arab', 'Emirates', '1996-12-06'),
                ('Japan', 'began', 'the', 'defence', 'of', 'their', 'Asian', 'Cup', 'title', 'with', 'a', 'lucky', 
                '2-1', 'win', 'against', 'Syria', 'in', 'a', 'Group', 'C', 'championship', 'match', 'on', 'Friday', '.')
            ], 
            'tags': [
                ('I-LOC', 'O', 'I-LOC', 'I-LOC', 'I-LOC', 'O'),
                ('I-LOC', 'O', 'O', 'O', 'O', 'O', 'I-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I-LOC',
                'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O')
            ]
        },
        'pos-ud-ewt': {
            'sentences': [
                ('What', 'if', 'Google', 'Morphed', 'Into', 'GoogleOS', '?'),
                ('What', 'if', 'Google', 'expanded', 'on', 'its', 'search', '-', 'engine', '(', 'and', 'now', 'e-mail', ')',
                'wares', 'into', 'a', 'full', '-', 'fledged', 'operating', 'system', '?')
            ],
            'tags': [
                ('PRON', 'SCONJ', 'PROPN', 'VERB', 'ADP', 'PROPN', 'PUNCT'),
                ('PRON', 'SCONJ', 'PROPN', 'VERB', 'ADP', 'PRON', 'NOUN', 'PUNCT', 'NOUN', 'PUNCT', 'CCONJ',
                'ADV', 'NOUN', 'PUNCT', 'NOUN', 'ADP', 'DET', 'ADV', 'PUNCT', 'ADJ', 'NOUN', 'NOUN', 'PUNCT')
            ]
        }
    }

    return expected_dict


@pytest.fixture
def dummy_tokenizer():
    dummy_data = {
        'train': (
            [('add', 'sabrina', 'salerno', 'to', 'the', 'grime', 'instrumentals', 'playlist')],
            [('O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O')]
        ),
        'test': (
            [('what', 's', 'the', 'weather', 'here', 'on', '2/7/2021')],
            [('O', 'O', 'O', 'O', 'B-current_location', 'O', 'B-timeRange')]
        )
    }

    token2idx = {
        'PADDING': 0, 'UNK': 1, 'NULL': 2, 'add': 3, 'sabrina': 4, 'salerno': 5, 'to': 6, 'the': 7,
        'grime': 8, 'instrumentals': 9, 'playlist': 10
    }

    label2idx = {
        'PADDING': 0, 'O': 1, 'B-artist': 2, 'I-artist': 3, 'B-playlist': 4, 'I-playlist': 5,
        'B-current_location': 6, 'B-timeRange': 7
    }

    dummy_data_tensor = {
        'train': [
            torch.cat((torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long),
                       torch.zeros(52, dtype=torch.long))),
            torch.cat((torch.tensor([1, 2, 3, 1, 1, 4, 5, 1], dtype=torch.long),
                       torch.zeros(52, dtype=torch.long)))
        ],
        'test': [
            torch.cat((torch.tensor([1, 1, 7, 1, 1, 1, 1], dtype=torch.long),
                       torch.zeros(53, dtype=torch.long))),
            torch.cat((torch.tensor([1, 1, 1, 1, 6, 1, 7], dtype=torch.long),
                       torch.zeros(53, dtype=torch.long))),
        ]
    }

    return dummy_data, token2idx, label2idx, dummy_data_tensor
