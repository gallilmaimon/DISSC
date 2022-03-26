import numpy as np
from pathlib import Path

def data_split(data_path: str, split_method: str = 'random', train_size:float = .7) -> None:
    if split_method == 'random':
        base_path = Path(data_path).parent.absolute()
        with open(data_path, 'r') as f, open(base_path / 'train.txt', 'a+') as f_tr, \
                open(base_path / 'val.txt', 'a+') as f_val:
            for line in f.readlines():
                if np.random.rand() <= train_size:
                    f_tr.write(line)
                else:
                    f_val.write(line)
    else:
        raise f"Unsupported train-val split method {split_method}"

def calculate_pitch_stats(data_path: str) -> None:
    pass