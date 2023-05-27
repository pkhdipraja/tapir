import torch
import numpy as np

from collections import defaultdict
from model.model_utils import rnn_add_null_tokens


def get_partial_output(cfgs, loader, model, token2idx):
    """
    Get incremental outputs for non-recurrent model.
    """
    results = defaultdict(dict)
    model.eval()

    with torch.no_grad():
        for idx, (seq, tag) in enumerate(loader):  # We use batch size of 1
            labels_mask = (tag != 0)
            seq_len = torch.sum(labels_mask).item()
            active_tokens = torch.zeros(seq.size(1), dtype=torch.long).unsqueeze(0)
            active_mask = torch.zeros(seq.size(1), dtype=torch.bool).unsqueeze(0)

            # To store increasing prefix
            predictions = np.empty((seq_len, seq_len))
            predictions.fill(np.inf)

            # To store edits
            changes = np.zeros((seq_len, seq_len))

            # Split sequence into partial inputs:
            for length in range(1, seq_len+1):
                active_tokens[:, length-1] = seq[:, length-1]
                active_mask[:, length-1] = True

                if cfgs.MODEL == 'incremental-transformers':
                    out = model(active_tokens, valid=True)
                else:
                    out = model(active_tokens)
                out = torch.argmax(out, dim=2)

                # Save partial outputs
                predictions[length-1][:length] = out[active_mask].numpy()

                if length == 1:
                    changes[length-1][0] = 1
                else:
                    changes[length-1] = predictions[length-1] != predictions[length-2]

            active_tag = tag[labels_mask].view(1, -1)
            accuracy = (predictions[-1] == active_tag.numpy()).sum() / seq_len

            results['partial_outputs'][idx] = predictions
            results['log_changes'][idx] = changes
            results['accuracy'][idx] = accuracy

    return results


def get_partial_output_two_pass(cfgs, loader, model, token2idx):
    """
    Get incremental outputs for two-pass model.
    """
    results = defaultdict(dict)
    model.eval()

    with torch.no_grad():
        for idx, (seq, tag) in enumerate(loader):  # We use batch size of 1
            labels_mask = (tag != 0)
            seq_len = torch.sum(labels_mask).item()

            # To store increasing prefix
            predictions = np.empty((seq_len, seq_len))
            predictions.fill(np.inf)

            predictions_monotonic = np.empty((seq_len, seq_len))
            predictions_monotonic.fill(np.inf)

            # To store edits
            changes = np.zeros((seq_len, seq_len))

            # Add null tokens for TwoPass model with delayed output.
            if cfgs.DELAY > 0:
                seq = rnn_add_null_tokens(seq,
                                          cfgs.DELAY,
                                          model.token2idx['NULL'])

            # Split sequence into partial inputs
            for length in range(1, seq_len+1):
                active_tokens = seq[:, 0:length+cfgs.DELAY]

                out, rev = model(active_tokens, valid=True)
                out = torch.argmax(out, dim=2)

                # Output of pure incremental model
                incr_out = torch.argmax(model.model.incr_out, dim=2)
                predictions_monotonic[length-1][:length] = incr_out.numpy()

                # Save partial outputs
                predictions[length-1][:length] = out[:, cfgs.DELAY:].numpy()

                if length == 1:
                    changes[length-1][0] = 1
                else:
                    changes[length-1] = predictions[length-1] != predictions[length-2]

            active_tag = tag[labels_mask].view(1, -1)
            accuracy = (predictions[-1] == active_tag.numpy()).sum() / seq_len

            results['partial_outputs'][idx] = predictions
            results['log_changes'][idx] = changes
            results['accuracy'][idx] = accuracy
            results['monotonic_outputs'][idx] = predictions_monotonic
            results['revision_outputs'][idx] = (rev > cfgs.REV_THRESHOLD).numpy()
            results['input'][idx] = seq.numpy()
            results['token_idx'] = model.idx2token
            results['label_idx'] = model.idx2label
            results['gold_label'][idx] = tag.numpy()

    return results
