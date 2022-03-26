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