import torch
import numpy as np
from itertools import groupby


def get_spkrs_dict(path: str) -> dict:
    speakers = []
    with open(path, 'r') as f:
        for line in f.readlines():
            val_dict = eval(line)
            speakers.append(val_dict['audio'].split('_')[0])
    return {n:i for i, n in enumerate(np.unique(speakers))}

def dedup_seq(seq):
    vals, counts = zip(*[(k, sum(1 for _ in g)) for k,g in groupby(seq)])
    return vals, counts

def prep_stats_tensors(spk_id_dict, f0_param_dict):
    id2pitch_mean = torch.empty(len(spk_id_dict), requires_grad=False)
    id2pitch_std = torch.empty(len(spk_id_dict), requires_grad=False)
    for n, v in spk_id_dict.items():
        stats = f0_param_dict[n]
        id2pitch_mean[v] = stats['mean']
        id2pitch_std[v] = stats['std']

    return id2pitch_mean, id2pitch_std