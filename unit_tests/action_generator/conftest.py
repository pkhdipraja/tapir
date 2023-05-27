import numpy as np
import pytest

from collections import OrderedDict


class DummyConfig(object):
    def __init__(self):
        self.DATASET = None
        self.TASK_TYPE = 'labelling'

        self.SPLIT = {
            'train': 'train'
        }


@pytest.fixture
def expected_data():
    partial_outputs = OrderedDict({
        0:  np.array([
                [1, np.inf, np.inf, np.inf, np.inf],
                [1, 2, np.inf, np.inf, np.inf],
                [1, 2, 3, np.inf, np.inf],
                [1, 2, 3, 4, np.inf],
                [1, 2, 3, 4, 5]
            ]),
        1:  np.array([
                [1, np.inf, np.inf, np.inf, np.inf],
                [2, 3, np.inf, np.inf, np.inf],
                [3, 4, 5, np.inf, np.inf, np.inf],
                [4, 5, 6, 7, np.inf],
                [5, 6, 7, 8, 9]
            ]),
        2:  np.array([
                [20, np.inf, np.inf, np.inf, np.inf],
                [312, 20, np.inf, np.inf, np.inf],
                [312, 20, 61, np.inf, np.inf],
                [312, 81, 61, 70, np.inf],
                [312, 81, 61, 70, 92]
            ]),
        3:  np.array([
                [15, np.inf, np.inf, np.inf, np.inf],
                [15, 18, np.inf, np.inf, np.inf],
                [20, 18, 17, np.inf, np.inf],
                [19, 18, 17, 87, np.inf],
                [19, 18, 17, 87, 98]
            ]),
        4:  np.array([
                [56, np.inf, np.inf, np.inf, np.inf],
                [74, 103, np.inf, np.inf, np.inf],
                [72, 109, 652, np.inf, np.inf],
                [72, 109, 652, 986, np.inf],
                [72, 109, 652, 986, 1032]
            ])
    })

    expected_actions = OrderedDict({
        0: np.zeros((5), dtype=np.int8),
        1: np.array([
                0, 1, 1, 1, 1
            ], dtype=np.int8),
        2: np.array([
                0, 1, 0, 1, 0
            ], dtype=np.int8),
        3: np.array([
                0, 0, 1, 1, 0
            ], dtype=np.int8),
        4: np.array([
                0, 1, 1, 0, 0
            ], dtype=np.int8)
    })

    return partial_outputs, expected_actions
