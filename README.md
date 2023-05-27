## TAPIR: Learning Adaptive Revision for Incremental Natural Language Understanding with a Two-Pass Model

This repository contains the code for [TAPIR: Learning Adaptive Revision for Incremental Natural Language Understanding with a Two-Pass Model](https://arxiv.org/abs/2305.10845).

### Setup
* Install python3 requirements: `pip install -r requirements.txt`
* Initialize GloVe as follows:
```bash
$ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.3.0/en_vectors_web_lg-2.3.0.tar.gz -O en_vectors_web_lg-2.3.0.tar.gz
$ pip install en_vectors_web_lg-2.3.0.tar.gz
```

### Action Generation
The script `gen_actions.py` can be used to generate the action sequences required for training TAPIR, being saved under `dataset_action/`:

```bash
$ python3 gen_actions.py --RUN train --MODEL_CONFIG <model_config> --DATASET <dataset> --CKPT_V <model_version> --CKPT_E <model_epoch> --GEN_SPLIT <train_only/valid>
```

Note: Please uncomment L58 of `configs/config.py` before running `gen_actions.py`

### Training
You should first create a model configuration file under `configs/` (see the provided sample). The following script will run the training:
```bash
$ python3 main.py --RUN train --MODEL_CONFIG <model_config> --DATASET <dataset>
```

with checkpoint saved under `ckpts/<dataset>/` and log under `results/log/`

Important parameters:
1. `--VERSION str`, to assign a name for the model.
2. `--GPU str`, to train the model on specified GPU. For multi-GPU training, use e.g. `--GPU '0, 1, 2, ...'`.
3. `--SEED int`, set seed for this experiment.
4. `--RESUME True`, start training with saved checkpoint. You should assign checkpoint version `--CKPT_V str` and resumed epoch `--CKPT_E int`.
5. `--NW int`, to accelerate data loading speed.
6. `--DATA_ROOT_PATH str`, to set path to your dataset.

To check all possible parameters, use `--help`

### Testing
You can evaluate on validation or test set using `--RUN {val, test}`. For example:
```bash
$ python3 main.py --RUN test --MODEL_CONFIG <model_config> --DATASET <dataset> --CKPT_V <model_version> --CKPT_E <model_epoch>
```
or with absolute path:
```bash
$ python3 main.py --RUN test --MODEL_CONFIG <model_config> --DATASET <dataset> --CKPT_PATH <path_to_checkpoint>.ckpt
```

To obtain incremental evaluation on the test sets, use the flag `--INCR_EVAL`. For incremental inference benchmark, use `--SPD_BENCHMARK`

### Data
We do not upload the original datasets as some of them needs license agreements. It is possible to download them through the links below. The preprocessing steps are described in the paper and the appendix.

As it is, the code can run experiments on:

* Slot filling - SNIPS [link](https://github.com/snipsco/nlu-benchmark)
* Slot filling - Alarm, reminder & weather [link](https://fb.me/multilingual_task_oriented_data)
* Slot filling - MIT Movie [link](https://groups.csail.mit.edu/sls/downloads/movie/)
* Named Entity Recognition, Chunking, and PoS tagging - CoNLL-2003 [link](https://www.clips.uantwerpen.be/conll2003/ner/)
* PoS tagging - Universal Dependencies English Web Treebank [link](https://github.com/UniversalDependencies/UD_English-EWT)

Data has to be split into three files (`data/train/train.<task>`, `data/valid/valid.<task>` and `data/test/test.<task>`) as in `configs/path_config.yml`, all of them following the format:

```yml
token \t label \n token \t label \n
```
with an extra \n between sequences.

If the repository is helpful for your research, we would really appreciate if you could cite the paper:


```
@inproceedings{kahardipraja23,
    title = "{TAPIR}: {L}earning Adaptive Revision for Incremental Natural Language Understanding with a Two-Pass Model",
    author = "Kahardipraja, Patrick  and
      Madureira, Brielen  and
      Schlangen, David",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "[To Appear]",
    pages = "[To Appear]",
}
```