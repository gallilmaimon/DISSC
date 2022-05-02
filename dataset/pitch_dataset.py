import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# constants
EPS = 0.001

class PitchDataset(Dataset):
    def __init__(self, path, spk_id_dict, f0_param_dict, nbins=50, f_min=None, scale=None):
        self.spk_id_dict = spk_id_dict
        self.f0_param_dict = f0_param_dict
        self.vals, self.fs, self.spk_ids = self._prepare_dataset(path)
        self.nbins = nbins
        self.f_min, self.scale = self._get_scaling(f_min, scale)  # transition to smoothed one-hot is on loss for memory reasons

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i], self.fs[i], self.spk_ids[i]

    def _prepare_dataset(self, path):
        fs, seqs, spk_ids = [], [], []
        with open(path, 'r') as f:
            for line in f.readlines():
                val_dict = eval(line)
                name = val_dict['audio'].split('_')[0]
                seqs.append(torch.IntTensor(val_dict['units']))
                fs.append((torch.FloatTensor(val_dict['f0']) - self.f0_param_dict[name]['mean']) / self.f0_param_dict[name]['std'])
                spk_ids.append(torch.IntTensor([self.spk_id_dict[name]]))
        return pad_sequence(seqs, batch_first=True, padding_value=100), pad_sequence(fs, batch_first=True,
                                                                                     padding_value=100), torch.concat(
            spk_ids).view(-1, 1)

    def _get_scaling(self, f_min=None, scale=None):
        if f_min is None:
            f_min = self.fs.min()
        if scale is None:
            scale = (self.fs.max() + EPS - f_min) / self.nbins
        return f_min, scale