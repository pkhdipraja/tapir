import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import model.linear_transformer as linear_transformer
import model.transformer as transformer

from model.incremental import RevisionModel
from model.model_utils import add_null_tokens, rnn_add_null_tokens
from collections import OrderedDict
from seqeval.metrics import f1_score
from torch.optim.lr_scheduler import MultiStepLR


class LinearTransformerLabellingBase(pl.LightningModule):
    """
    Linear Transformer base module for sequence labelling.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb, position_enc, encoder):
        super(LinearTransformerLabellingBase, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.token2idx = token2idx
        self.label2idx = label2idx
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = encoder(
            cfgs, self.token_size,
            pretrained_emb, position_enc
        )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.out_proj(self.encoder(x))
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({
            'val_loss': loss,
            'val_acc': val_acc,
            'val_pred': pred,
            'val_label': active_targets
        })

        return output

    def test_step(self, batch, batch_idx):
        # This assume batch size = 1
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({
            'test_loss': loss,
            'test_acc': test_acc,
            'test_pred': pred,
            'test_label': active_targets
        })

        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(
                optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE
            ),
            'name': 'LR scheduler'
        }

        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class LinearCausalEncoderLabelling(LinearTransformerLabellingBase):
    """
    Linear Transformers with causal masking for sequence labelling task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        encoder = linear_transformer.LinearCausalEncoderLabelling
        super(LinearCausalEncoderLabelling, self).__init__(
            cfgs, token2idx, label2idx, pretrained_emb,
            position_enc, encoder
        )


class LinearEncoderLabelling(LinearTransformerLabellingBase):
    """
    Linear Transformers for sequence labelling task.
    Attend to all tokens.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        encoder = linear_transformer.LinearEncoderLabelling
        super(LinearEncoderLabelling, self).__init__(
            cfgs, token2idx, label2idx, pretrained_emb,
            position_enc, encoder
        )


class TransformerBase(pl.LightningModule):
    """
    Standard Transformer base module.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb, position_enc, encoder):
        super(TransformerBase, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.token2idx = token2idx
        self.label_size = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = encoder(
            cfgs, self.token_size,
            pretrained_emb, position_enc
        )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)

    def forward(self, x):
        logits = self.out_proj(self.encoder(x))
        return logits

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(
                optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE
            ),
            'name': 'LR scheduler'
        }

        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class TransformerEncoderLabelling(TransformerBase):
    """
    Standard Transformer encoder for sequence
    labelling task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        encoder = transformer.EncoderLabelling
        super(TransformerEncoderLabelling, self).__init__(
            cfgs, token2idx, label2idx, pretrained_emb,
            position_enc, encoder
        )
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({'val_loss': loss,
                              'val_acc': val_acc,
                              'val_pred': pred,
                              'val_label': active_targets})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({'test_loss': loss,
                              'test_acc': test_acc,
                              'test_pred': pred,
                              'test_label': active_targets})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)


class IncrementalTransformerEncoderLabelling(TransformerBase):
    """
    Standard incremental Transformer encoder for sequence
    labelling task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        encoder = transformer.IncrementalEncoderLabelling
        super(IncrementalTransformerEncoderLabelling, self).__init__(
            cfgs, token2idx, label2idx, pretrained_emb,
            position_enc, encoder
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, valid=False):
        logits = self.out_proj(self.encoder(x, valid))
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({'val_loss': loss,
                              'val_acc': val_acc,
                              'val_pred': pred,
                              'val_label': active_targets})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({'test_loss': loss,
                              'test_acc': test_acc,
                              'test_pred': pred,
                              'test_label': active_targets})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)


class TwoPassLabelling(pl.LightningModule):
    """
    Revision model module for sequence labelling.
    """
    def __init__(self, cfgs, token2idx, label2idx, reviser,
                 pretrained_emb=None, position_enc=True):
        super(TwoPassLabelling, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.token2idx = token2idx
        self.label2idx = label2idx
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.idx2token = {value: key for key, value in token2idx.items()}
        self.reviser = reviser(
            cfgs, token2idx, label2idx,
            pretrained_emb, position_enc
        )
        self.reviser.eval()
        self.model = RevisionModel(
            cfgs, self.token_size, self.label_size,
            self.reviser, pretrained_emb
        )
        self.loss_enc = nn.CrossEntropyLoss()
        self.loss_ctrl = nn.BCELoss()

    def forward(self, x, valid=False):
        return self.model(x, valid)

    def training_step(self, batch, batch_idx):
        inputs, targets, rev_targets = batch

        if self.cfgs.DELAY > 0:
            inputs = rnn_add_null_tokens(inputs,
                                         self.cfgs.DELAY,
                                         self.token2idx['NULL'])
            targets = F.pad(targets, (self.cfgs.DELAY, 0), "constant", 0)
            rev_targets = F.pad(rev_targets, (self.cfgs.DELAY, 0), "constant", 0)  # Force the model to WRITE during initial step

        logits, rev = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_revision_mask = (rev_targets != -1)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]
        active_rev = rev[active_revision_mask]
        active_rev_targets = rev_targets[active_revision_mask]

        loss = self.loss_enc(active_logits, active_targets) + self.loss_ctrl(active_rev, active_rev_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # This assume batch size = 1
        inputs, targets = batch

        if self.cfgs.DELAY > 0:
            inputs = rnn_add_null_tokens(inputs,
                                         self.cfgs.DELAY,
                                         self.token2idx['NULL'])
            targets = F.pad(targets, (self.cfgs.DELAY, 0), "constant", 0)

        logits, rev = self.forward(inputs, valid=True)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss_enc(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({
            'val_loss': loss,
            'val_acc': val_acc,
            'val_pred': pred,
            'val_label': active_targets
        })

        return output

    def test_step(self, batch, batch_idx):
        # This assume batch size of 1
        inputs, targets = batch

        if self.cfgs.DELAY > 0:
            inputs = rnn_add_null_tokens(inputs,
                                         self.cfgs.DELAY,
                                         self.token2idx['NULL'])
            targets = F.pad(targets, (self.cfgs.DELAY, 0), "constant", 0)

        logits, rev = self.forward(inputs, valid=True)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss_enc(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({
            'test_loss': loss,
            'test_acc': test_acc,
            'test_pred': pred,
            'test_label': active_targets
        })

        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(
                optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE
            ),
            'name': 'LR scheduler'
        }

        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)
