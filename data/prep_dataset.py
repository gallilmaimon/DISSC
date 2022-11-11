import argparse
from data_utils import data_split, calculate_pitch_stats
from utils import seed_everything


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoded_path', default='ESD/hubert100/train.txt', help='Path for HuBERT encodings')
    parser.add_argument('--stats_path', default='ESD/hubert100/f0_stats.pkl', help='Output path for train speaker stats')
    parser.add_argument('--seed', default=42, type=int, help='number of unique HuBERT clusters to used')
    parser.add_argument('--split_method', default=None, help='Method for train-test split. If None encoded path is all train and no split is performed')

    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(args.seed)
    if args.split_method:
        train_path, _ = data_split(args.encoded_path, split_method=args.split_method)
    else:
        train_path = args.encoded_path
    calculate_pitch_stats(train_path, args.stats_path)
