# Copyright 2019 Vision and Language Group@ MIL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------------------------------------------
# Adapted from OpenVQA (https://github.com/MILVLG/openvqa/blob/master/openvqa/core/path_cfgs.py)
# Originally written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os
import yaml
import logging

from collections import defaultdict


class PATH(object):
    """
    Configuration object for dataset paths.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Datasets root path
        self.DATA_ROOT_PATH = '/home/users/data/'
        self.DATA_PATH = defaultdict(dict)

    def init_path(self, path_dicts):
        self.SPLIT_PATH = {
            'train': self.DATA_ROOT_PATH + 'train/',
            'valid': self.DATA_ROOT_PATH + 'valid/',
            'test': self.DATA_ROOT_PATH + 'test/'
        }

        for dataset in path_dicts:
            for item in path_dicts[dataset]:
                self.DATA_PATH[dataset][item] = self.SPLIT_PATH[item] + \
                    path_dicts[dataset][item]

        self.RESULT_PATH = './results/result_test'
        self.LOG_PATH = './results/log'
        self.CKPTS_PATH = './ckpts'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir(self.RESULT_PATH)

        if 'log' not in os.listdir('./results'):
            os.mkdir(self.LOG_PATH)

        if 'ckpts' not in os.listdir('./'):
            os.mkdir(self.CKPTS_PATH)

    def check_path(self, dataset=None):
        self.logger.info("Checking the datasets...")

        if dataset:
            for item in self.DATA_PATH[dataset]:
                if not os.path.exists(self.DATA_PATH[dataset][item]):
                    self.logger.error(self.DATA_PATH[dataset][item] + 'MISSING')
                    exit(-1)
        else:
            # Check if all datasets exist
            for dataset in self.DATA_PATH:
                for item in self.DATA_PATH[dataset]:
                    if not os.path.exists(self.DATA_PATH[dataset][item]):
                        self.logger.error(self.DATA_PATH[dataset][item] + 'MISSING')
                        exit(-1)

        self.logger.info("Finished checking the datasets!")
