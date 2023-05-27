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
# Adapted from OpenVQA (https://github.com/MILVLG/openvqa/blob/master/openvqa/core/base_cfgs.py)
# Originally written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import os
import random
import torch

from configs.path_configs import PATH
from types import MethodType


class ExpConfig(PATH):
    """
    Configuration object for model & experiments.
    """
    def __init__(self):
        super(ExpConfig, self).__init__()

        # Set devices
        # For multi-gpu, set e.g. '0, 1, 2' instead
        self.GPU = '0'

        # Set RNG for CPU and GPU
        self.SEED = random.randint(0, 99999999)

        # Define a random seed for new training
        self.VERSION = 'default'

        # For resuming training and testing
        self.CKPT_VERSION = self.VERSION + '_' + str(self.SEED)
        self.CKPT_EPOCH = 0

        # Absolute checkpoint path, override 'CKPT_VERSION' and 'CKPT_EPOCH
        self.CKPT_PATH = None

        # Set training split
        self.TRAIN_SPLIT = 'train'

        # Define data split
        self.SPLIT = {
            'train': '', 'valid': 'valid', 'test': 'test'  #, 'train_only': 'train' -> only turn this on for gen_actions
        }

        # Define maximum sentence length
        self.MAX_TOKEN = 200

        # Use GloVe embeddings
        self.USE_GLOVE = True

        # Word embeddings size
        self.WORD_EMBED_SIZE = 300

        # Early stopping patience
        self.PATIENCE = 10

        # Dataset list
        self.SEQ_LABELLING = ['snips-slot', 'multilingual', 'movie',
                              'chunk-conll2003', 'pos-conll2003', 'ner-conll2003',
                              'pos-ud-ewt']

        # Task that uses BIO scheme
        self.BIO_SCHEME = ['snips-slot', 'multilingual', 'movie', 'chunk-conll2003', 'ner-conll2003']

        # Optimizer
        self.OPT = ''
        self.OPT_PARAMS = {}

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def setup(self):
        def all(iterable):
            for element in iterable:
                if not element:
                    return False
            return True

        assert self.RUN_MODE in ['train', 'val', 'test'], "Please select a mode"

        # Ensure hyperparams for probability is between 0 and 1
        if self.DROPOUT < 0 or self.DROPOUT > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, got {}".format(self.DROPOUT))

        if self.UNK_PROB < 0 or self.UNK_PROB > 1:
            raise ValueError("Probability of replacing tokens with UNK has to be between 0 and 1, \
                got {}".format(self.UNK_PROB))

        # Ensure delay is not negative
        if self.DELAY < 0 or self.DELAY > 10:
            raise ValueError("Delay has to be between 0 and 10, got {}".format(self.DELAY))

        # ---------- Setup devices ----------
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)

        # ---------- Setup Opt ----------
        assert self.OPT in ['Adam', 'AdamW', 'RMSProp', 'SGD', 'Adagrad']
        optim = getattr(torch.optim, self.OPT)
        default_params_dict = dict(zip(optim.__init__.__code__.co_varnames[3: optim.__init__.__code__.co_argcount],
                                       optim.__init__.__defaults__[1:]))

        assert all(list(map(lambda x: x in default_params_dict, self.OPT_PARAMS)))

        for key in self.OPT_PARAMS:
            if isinstance(self.OPT_PARAMS[key], str):
                self.OPT_PARAMS[key] = eval(self.OPT_PARAMS[key])
            else:
                self.logger.error("To avoid ambiguity, set the value of 'OPT_PARAMS' to string type")
                exit(-1)
        self.OPT_PARAMS = {**default_params_dict, **self.OPT_PARAMS}

        if self.CKPT_PATH is not None:
            self.logger.warning("CKPT_VERSION will not work with CKPT_PATH")
            self.CKPT_VERSION = self.CKPT_PATH.split('/')[-1] + '_' + str(self.SEED)

        if self.CKPT_VERSION.split('_')[0] != self.VERSION and self.RUN_MODE in ['val', 'test']:
            self.VERSION = self.CKPT_VERSION

        # ---------- Setup split ----------
        self.SPLIT['train'] = self.TRAIN_SPLIT

        # ---------- Task type ----------
        if self.DATASET in self.SEQ_LABELLING:
            self.TASK_TYPE = 'labelling'
        # elif self.DATASET in self.SEQ_CLASSIFICATION:
        #     self.TASK_TYPE = 'classification'
        else:
            exit(-1)

    def config_dict(self):
        conf_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and not isinstance(getattr(self, attr), MethodType):
                conf_dict[attr] = getattr(self, attr)
        return conf_dict

    def __str__(self):
        for attr in dir(self):
            if not (attr.startswith('_') or attr == 'logger') and not isinstance(getattr(self, attr), MethodType):
                self.logger.info("%s -> %s", attr, getattr(self, attr))

        return ''
