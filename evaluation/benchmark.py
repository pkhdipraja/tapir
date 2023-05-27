import numpy as np
import torch


def speed_benchmark_baseline(cfgs, loader, model):
    """
    Get total runtime for baseline model (restart-incremental).
    """
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        for idx, (seq, tag) in enumerate(loader):  # We use batch size of 1
            labels_mask = (tag != 0)
            seq_len = torch.sum(labels_mask).item()
            active_tokens = torch.zeros(seq.size(1), dtype=torch.long).unsqueeze(0)

            for length in range(seq_len):
                active_tokens[:, length] = seq[:, length]
                out = model(active_tokens)
                out = torch.argmax(out, dim=2)

        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)/1000

    return elapsed_time


def speed_benchmark_twopass(cfgs, loader, model):
    """
    Get total runtime for two pass model.
    """
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        for idx, (seq, tag) in enumerate(loader):  # We use batch size of 1
            out, _ = model(seq, valid=True)
            out = torch.argmax(out, dim=2)
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)/1000

    return elapsed_time


def speed_benchmark(cfgs, loader, model):
    if cfgs.MODEL == 'two-pass':
        elapsed = speed_benchmark_twopass(cfgs, loader, model)
    else:
        elapsed = speed_benchmark_baseline(cfgs, loader, model)

    return len(loader)/elapsed
