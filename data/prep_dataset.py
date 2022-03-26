from data_utils import data_split

if __name__ == '__main__':
    data_split('VCTK-corpus/hubert100/encoded.txt', split_method='random', train_size=.7)