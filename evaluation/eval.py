# MIT License

# Copyright (c) 2020 briemadu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# --------------------------------------------------------
# Adapted from https://github.com/briemadu/inc-bidirectional/blob/master/structs.py
# Originally written by Brielen Madureira https://github.com/briemadu
# --------------------------------------------------------

import numpy as np
import itertools
import torch

from utils.partial_utils import get_partial_output, get_partial_output_two_pass
from model.model_module import TransformerEncoderLabelling, TwoPassLabelling


class IncrementalMetrics(object):
    """
    Stores partial inputs and evaluate with incremental metrics.
    """
    def __init__(self, cfgs, loader, model, token2idx):
        self.cfgs = cfgs
        self.model = model
        cfgs.logger.info('Getting incremental results for: {}'.format(cfgs.MODEL))

        if cfgs.MODEL == 'two-pass':
            self.results = get_partial_output_two_pass(cfgs, loader, model, token2idx)
        else:
            self.results = get_partial_output(cfgs, loader, model, token2idx)

        self.seq_len = {key: value.shape[0] for key, value in
                        self.results['partial_outputs'].items()}

        self.estimate_edit_overheads()
        self.estimate_correction_times()
        self.estimate_relative_correctness()

        self.perc_accurate = len([x for x in
                                  self.results['accuracy'].values() if x == 1]) / len(self.results['accuracy'])

    def stats(self, metric_dict, only_correct=False, only_incorrect=False):
        """
        Estimates mean and standard deviation of values in a dictionary.
        """

        if only_correct:
            mean = np.mean([
                value for key, value in metric_dict.items()
                if self.results['accuracy'][key] == 1
            ])

            std = np.std([
                value for key, value in metric_dict.items()
                if self.results['accuracy'][key] == 1
            ])

        elif only_incorrect:
            mean = np.mean([
                value for key, value in metric_dict.items()
                if self.results['accuracy'][key] != 1
            ])

            std = np.std([
                value for key, value in metric_dict.items()
                if self.results['accuracy'][key] != 1
            ])
        else:
            mean = np.mean(list(metric_dict.values()))
            std = np.std(list(metric_dict.values()))

        return mean, std

    def estimate_edit_overheads(self):
        """
        Creates dictionaries of edit overhead (EO), delay of t={0,1,2}.
        EO is the number of unnecessary edits (subtitutions) divided by
        the total number edits (addition + subtitutions).
        """

        self.edit_overhead = {}
        self.edit_overhead_d1 = {}
        self.edit_overhead_d2 = {}

        for idx, changes in self.results['log_changes'].items():
            self.edit_overhead[idx] = self._get_edit_overhead(changes)
            self.edit_overhead_d1[idx] = self._get_edit_overhead_d1(changes)
            self.edit_overhead_d2[idx] = self._get_edit_overhead_d2(changes)

    def _get_edit_overhead(self, changes):
        necessary_additions = changes.diagonal().sum()
        subtitutions = (changes.sum() - changes.diagonal().sum())
        return subtitutions / (subtitutions + necessary_additions)

    def _get_edit_overhead_d1(self, changes):
        # sentences with 1 token
        if changes.shape[0] == 1:
            return 0

        # for sentence classification, the necessary addition is always 1
        if changes.shape[1] == 1:
            necessary_additions = 1
        else:
            # for sentence tagging, it's the number of tokens - 1
            # (we consider the two tokens on the last step as a single addition)
            necessary_additions = changes.diagonal().sum() - 1

        subtitutions = (changes.sum() - changes.diagonal().sum() - changes.diagonal(-1).sum())
        return subtitutions / (subtitutions + necessary_additions)

    def _get_edit_overhead_d2(self, changes):
        # sentences with 1 or two tokens
        if changes.shape[0] in (1, 2):
            return 0

        # for sentence classification, the necessary addition is always 1
        if changes.shape[1] == 1:
            necessary_additions = 1
        else:
            # for sentence tagging, it's the number of tokens - 2
            # (we consider the three tokens on the last step as a single addition)
            necessary_additions = changes.diagonal().sum() - 2

        subtitutions = (changes.sum() - changes.diagonal().sum() - changes.diagonal(-1).sum() - \
                        changes.diagonal(-2).sum())
        return subtitutions / (subtitutions + necessary_additions)

    def estimate_correction_times(self):
        """
        Creates dictionaries of correction time score (CT), delay of t={0,1,2}.
        CTScore is a score of the sum of the number of steps it took for a final
        decision to be reached for each label, divided by the number of all
        possible steps.
        """
        self.correction_time_pertimestep = {}
        self.correction_time_score = {}

        for idx, outputs in self.results['partial_outputs'].items():
            ct = self._get_correction_times(outputs)
            ct_len = len(ct)
            self.correction_time_pertimestep[idx] = ct

            if ct_len == 1:
                self.correction_time_score[idx] = 0
            else:
                self.correction_time_score[idx] = np.sum(ct) / (((ct_len - 1) * ct_len)/2)  # (len-1)+(len-2)+...+1

    def _get_correction_times(self, outputs):
        seq_len = outputs.shape[0]
        FD = []  # Final decision, FO is the index of the input
        for c, column in enumerate(outputs.T):
            # Final seq, correct input was chosen and did not change anymore
            last_group = [tuple(g) for _, g in itertools.groupby(column)][-1]
            # Time step when final decision was made
            # In other words, necessary steps to get the correct label
            # 0 means no change happened
            FD.append((seq_len - c) - len(last_group))

        return FD

    def estimate_relative_correctness(self):
        """
        Creates dictionaries of relative correctness (RC), delay of t={0,1,2}.
        RC is the proportion of partial inputs that are correct w.r.t the final, 
        non-incremental output.
        """
        self.relative_correctness = {}
        self.relative_correctness_d1 = {}
        self.relative_correctness_d2 = {}

        for idx, outputs in self.results['partial_outputs'].items():
            self.relative_correctness[idx] = self._get_relative_correctness(outputs)
            self.relative_correctness_d1[idx] = self._get_relative_correctness(outputs, delay=1)
            self.relative_correctness_d2[idx] = self._get_relative_correctness(outputs, delay=2)

    def _get_relative_correctness(self, outputs, delay=0):
        step = 1

        if delay == 1:
            if outputs.shape[0] == 1:
                return 1
            step = 0
        elif delay == 2:
            if outputs.shape[0] in [1, 2]:
                return 1
            step = -1

        correct_guesses = [
            np.array_equal(outputs[i][:i+step], outputs[-1][:i+step]) for i in range(delay, outputs.shape[0])
        ]

        r_correctness = np.mean(correct_guesses)
        return r_correctness

    def print_metrics(self, logger=None):
        mean_eo, std_eo = self.stats(self.edit_overhead)
        mean_ct, std_ct = self.stats(self.correction_time_score)
        mean_rc, std_rc = self.stats(self.relative_correctness)
        mean_eo_d1, std_eo_d1 = self.stats(self.edit_overhead_d1)
        mean_eo_d2, std_eo_d2 = self.stats(self.edit_overhead_d2)
        mean_rc_d1, std_rc_d1 = self.stats(self.relative_correctness_d1)
        mean_rc_d2, std_rc_d2 = self.stats(self.relative_correctness_d2)

        self.cfgs.logger.info("Mean EO: {}".format(mean_eo))
        self.cfgs.logger.info("Mean CT: {}".format(mean_ct))
        self.cfgs.logger.info("Mean RC: {}".format(mean_rc))

        if logger:
            if isinstance(self.model, TransformerEncoderLabelling):
                logger.experiment.log_metric('mean_EO', mean_eo)
                logger.experiment.log_metric('mean_CT', mean_ct)
                logger.experiment.log_metric('mean_RC', mean_rc)
                logger.experiment.log_metric('mean_EO_1', mean_eo_d1)
                logger.experiment.log_metric('mean_RC_1', mean_rc_d1)
                logger.experiment.log_metric('mean_EO_2', mean_eo_d2)
                logger.experiment.log_metric('mean_RC_2', mean_rc_d2)
            elif isinstance(self.model, TwoPassLabelling):
                if self.cfgs.DELAY == 0:
                    logger.experiment.log_metric('mean_EO', mean_eo)
                    logger.experiment.log_metric('mean_CT', mean_ct)
                    logger.experiment.log_metric('mean_RC', mean_rc)
                elif self.cfgs.DELAY == 1:
                    # This is due to the NULL token implementation, not a typo.
                    logger.experiment.log_metric('mean_EO_1', mean_eo)
                    logger.experiment.log_metric('mean_CT', mean_ct)
                    logger.experiment.log_metric('mean_RC_1', mean_rc)
                elif self.cfgs.DELAY == 2:
                    # This is due to the NULL token implementation, not a typo.
                    logger.experiment.log_metric('mean_EO_2', mean_eo)
                    logger.experiment.log_metric('mean_CT', mean_ct)
                    logger.experiment.log_metric('mean_RC_2', mean_rc)
