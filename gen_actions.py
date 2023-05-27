import numpy as np
import argparse
import logging
import os
import torch
import yaml

from utils.partial_utils import get_partial_output
from utils.utils import dm_dict, model_dict
from dataset_readers.datasets import Loader
from configs.config import ExpConfig
from pytorch_lightning import Trainer
from collections import defaultdict


class ActionSeq(object):
    """
    Stores partial outputs and generates WRITE/REVISE
    action sequences.
    """
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.results = defaultdict(dict)
        self.result_path = "./dataset_action/"

    def gen_partial_outputs(self, loader, model, token2idx):
        self.results = get_partial_output(
            self.cfgs, loader, model, token2idx
        )

    def gen_actions(self):
        data_size = len(self.results['partial_outputs'].keys())

        for idx in range(data_size):
            partial_output = self.results['partial_outputs'][idx]
            seq_len = partial_output.shape[0]

            actions = np.zeros((seq_len), dtype=np.int8)

            # First action is always WRITE
            for step in range(1, seq_len):
                if np.any(partial_output[step][:step] != partial_output[step-1][:step]):
                    actions[step] = 1  # mark the action as REVISE

            self.results['actions'][idx] = actions

    def gen_datasets(self, split='train_only'):
        dataset_id = self.cfgs.DATASET
        dataloader = Loader(self.cfgs)
        data = dataloader.load()

        with open(os.path.join(self.result_path, dataset_id + '_' + split + '_actions'), 'w') as f:
            sentence_list, tag_list = data[split]
            for idx, (seq_iter, tag_iter) in enumerate(zip(sentence_list, tag_list)):
                actions_iter = self.results['actions'][idx].tolist()
                for seq, tag, action in zip(seq_iter, tag_iter, actions_iter):
                    f.write("{}\t{}\t{}\n".format(seq, tag, action))

                f.write("\n")

    def compute_statistics(self):
        """
        Calculate average of WRITE and REVISE operation.
        """
        data_size = len(self.results['actions'].keys())
        seq_revise_statistics = defaultdict(int)
        revise_rate = 0.0
        write_rate = 0.0

        for idx in self.results['actions']:
            current_action = self.results['actions'][idx]
            seq_len = current_action.shape[0]

            seq_revise = np.sum(current_action == 1).item() / seq_len
            seq_write = np.sum(current_action == 0).item() / seq_len

            revise_rate += seq_revise
            write_rate += seq_write

            if seq_revise <= 0.2:
                seq_revise_statistics[0] += 1
            elif seq_revise > 0.2 and seq_revise <= 0.4:
                seq_revise_statistics[1] += 1
            elif seq_revise > 0.4 and seq_revise <= 0.6:
                seq_revise_statistics[2] += 1
            elif seq_revise > 0.6 and seq_revise <= 0.8:
                seq_revise_statistics[3] += 1
            else:
                seq_revise_statistics[4] += 1

        mean_revise_rate = revise_rate / data_size
        mean_write_rate = write_rate / data_size

        self.cfgs.logger.info("Mean revise rate: {}".format(mean_revise_rate))
        self.cfgs.logger.info("Mean write rate: {}".format(mean_write_rate))

        for i in range(len(seq_revise_statistics)):
            self.cfgs.logger.info("Seq revise count for bin {}: {}".format(i, seq_revise_statistics[i]))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Args for dataset action generation')

    parser.add_argument(
        '--RUN', dest='RUN_MODE',
        choices=['train', 'val', 'test'],
        help='{train, val, test}',
        type=str, required=True
    )

    parser.add_argument(
        '--MODEL_CONFIG', dest='MODEL_CONFIG',
        help='experiment configuration file',
        type=str, required=True
    )

    parser.add_argument(
        '--DATASET', dest='DATASET',
        choices=[
            'snips-slot', 'multilingual', 'movie',
            'chunk-conll2003', 'pos-conll2003', 'ner-conll2003', 'pos-ud-ewt'],
        help='{snips-slot, multilingual, movie, chunk-conll2003, pos-conll2003, \
            ner-conll2003, pos-ud-ewt}',
        type=str, required=True
    )

    parser.add_argument(
        '--CKPT_E', dest='CKPT_EPOCH',
        help='checkpoint epoch',
        type=int
    )

    parser.add_argument(
        '--CKPT_V', dest='CKPT_VERSION',
        help='checkpoint version',
        type=str
    )

    parser.add_argument(
        '--CKPT_PATH', dest='CKPT_PATH',
        help='load checkpoint path, if \
        possible use CKPT_VERSION and CKPT_EPOCH',
        type=str
    )

    parser.add_argument(
        '--NW', dest='NUM_WORKERS',
        help='multithreaded loading to accelerate IO',
        default=4,
        type=int
    )

    parser.add_argument(
        '--PINM', dest='PIN_MEM',
        help='disable pin memory',
        action='store_false',
    )

    parser.add_argument(
        '--SPLIT', dest='TRAIN_SPLIT',
        choices=['train', 'train+valid'],
        help='set training split',
        type=str
    )

    parser.add_argument(
        '--GEN_SPLIT', dest='GEN_SPLIT',
        choices=['train_only', 'valid'],
        help='set data generation split',
        type=str, required=True
    )

    args = parser.parse_args()
    return args


def main(cfgs):
    task = cfgs.TASK_TYPE
    cfgs.BATCH_SIZE = 1  # Only for action generation
    model_task = '_'.join([cfgs.MODEL, cfgs.TASK_TYPE])

    if cfgs.CKPT_PATH is not None:
        path = cfgs.CKPT_PATH
    else:
        path = os.path.join(cfgs.CKPTS_PATH, cfgs.DATASET,
                            '_'.join([
                                cfgs.DATASET, cfgs.MODEL,
                                cfgs.CKPT_VERSION,
                                'epoch=' + str(cfgs.CKPT_EPOCH) + '.ckpt'
                            ]))

    datamodule = dm_dict[task](cfgs)
    datamodule.prepare_data()
    datamodule.setup()

    if cfgs.USE_GLOVE:
            pretrained_emb = datamodule.tokenizer.pretrained_emb
    else:
        pretrained_emb = None

    model = model_dict[model_task]['test'](
                    cfgs, datamodule.tokenizer.token2idx,
                    datamodule.tokenizer.label2idx,
                    pretrained_emb
        )

    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])

    partial_buffer = ActionSeq(cfgs)

    if cfgs.GEN_SPLIT == 'train_only':
        partial_buffer.gen_partial_outputs(
            datamodule.train_dataloader(), model,
            datamodule.tokenizer.token2idx
        )
    else:
        partial_buffer.gen_partial_outputs(
            datamodule.val_dataloader(), model,
            datamodule.tokenizer.token2idx
        )

    cfgs.logger.info("Generating actions...")
    partial_buffer.gen_actions()
    cfgs.logger.info("Generating dataset...")
    partial_buffer.gen_datasets(split=cfgs.GEN_SPLIT)
    partial_buffer.compute_statistics()


if __name__ == "__main__":
    cfgs = ExpConfig()

    args = parse_args()
    args_dict = cfgs.parse_to_dict(args)

    path_cfg_file = './configs/path_config.yml'
    with open(path_cfg_file, 'r') as path_f:
        path_yaml = yaml.safe_load(path_f)

    model_cfg_file = './configs/{}.yml'.format(args.MODEL_CONFIG)
    with open(model_cfg_file, 'r') as model_f:
        model_yaml = yaml.safe_load(model_f)

    args_dict = {**args_dict, **model_yaml}

    cfgs.add_args(args_dict)
    cfgs.init_path(path_yaml)

    logging.basicConfig(level=logging.INFO)

    cfgs.setup()

    cfgs.logger.info("Hyperparameters:")
    cfgs.logger.info(cfgs)

    cfgs.check_path(dataset=args.DATASET)

    main(cfgs)
