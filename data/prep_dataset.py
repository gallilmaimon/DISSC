from data_utils import data_split, calculate_pitch_stats
from utils import seed_everything

if __name__ == '__main__':
    encoded_path = 'VCTK-corpus/hubert100/encoded.txt'
    stats_path = 'VCTK-corpus/hubert100/f0_stats.pkl'
    seed = 42

    if seed is not None:
        seed_everything(seed)
    train_path, _ = data_split(encoded_path, split_method='random', train_size=.7)
    calculate_pitch_stats(train_path, stats_path)
