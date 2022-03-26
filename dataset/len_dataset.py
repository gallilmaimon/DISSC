import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dataset.utils import dedup_seq


class LenDataset(Dataset):
    def __init__(self, path, spk_id_dict):
        self.vals, self.lens, self.spk_ids = LenDataset._prepare_dataset(path, spk_id_dict)

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i], self.lens[i], self.spk_ids[i]

    @staticmethod
    def _prepare_dataset(path, spk_id_dict):
        all_vals, all_counts, spk_ids = [], [], []
        with open(path, 'r') as f:
            for line in f.readlines():
                val_dict = eval(line)
                vals, counts = dedup_seq(val_dict['units'])
                all_vals.append(torch.IntTensor(vals))
                all_counts.append(torch.FloatTensor(counts))
                spk_ids.append(torch.IntTensor([spk_id_dict[val_dict['audio'].split('_')[0]]]))
        return pad_sequence(all_vals, batch_first=True, padding_value=100), \
               pad_sequence(all_counts, batch_first=True, padding_value=-1), torch.concat(spk_ids).view(-1, 1)