import pickle

import numpy as np
from pathlib import Path
from collections import defaultdict


def data_split(data_path: str, split_method: str = 'random', train_size:float = .7) -> (str, str):
    if split_method == 'random':
        base_path = Path(data_path).parent.absolute()
        with open(data_path, 'r') as f, open(base_path / 'train.txt', 'w') as f_tr, \
                open(base_path / 'val.txt', 'w') as f_val:
            for line in f.readlines():
                if np.random.rand() <= train_size:
                    f_tr.write(line)
                else:
                    f_val.write(line)
        return base_path / 'train.txt', base_path / 'val.txt'
    elif split_method == 'paired_val':
        base_path = Path(data_path).parent.absolute()
        with open(data_path, 'r') as f, open(base_path / 'train.txt', 'w') as f_tr, \
                open(base_path / 'val.txt', 'w') as f_val:
            for line in f.readlines():
                audio_num = int(eval(line)['audio'].split('_')[1])
                if audio_num <= 24:
                    f_val.write(line)
                else:
                    f_tr.write(line)
        return base_path / 'train.txt', base_path / 'val.txt'
    else:
        raise f"Unsupported train-val split method {split_method}"

def calculate_pitch_stats(data_path: str, out_path: str) -> None:
    speaker_fs = defaultdict(lambda :[])
    with open(data_path, 'r') as f:
        for line in f.readlines():
            val_dict = eval(line)
            speaker_fs[val_dict['audio'].split('_')[0]] += val_dict['f0']

    speaker_stats = dict()
    for k in speaker_fs.keys():
        speaker_fs_k = np.array(speaker_fs[k])[np.array(speaker_fs[k]) != 0]  # take only voiced parts
        speaker_stats[k] = {'mean': speaker_fs_k.mean(), 'std': speaker_fs_k.std()}

    with open(out_path, 'wb') as f_out:
        pickle.dump(speaker_stats, f_out)
