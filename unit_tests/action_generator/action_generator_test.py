import pytest
import numpy as np

from unit_tests.action_generator.conftest import DummyConfig, expected_data
from gen_actions import ActionSeq


def test_gen_actions(expected_data):
    cfgs = DummyConfig()
    partial_buffer = ActionSeq(cfgs)

    partial_outputs, expected_actions = expected_data
    partial_buffer.results['partial_outputs'] = partial_outputs

    partial_buffer.gen_actions()

    for action_iter in partial_buffer.results['actions'].keys():
        assert np.array_equal(
            partial_buffer.results['actions'][action_iter],
            expected_actions[action_iter]
        ), "Generated actions do not match expected values."
