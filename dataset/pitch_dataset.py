import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# constants
EPS = 0.001

class PitchDataset(Dataset):
    def __init__(self, path, spk_id_dict, f0_param_dict, n_tokens=100, padding_value=-100, normalise_pitch=True):
        self.spk_id_dict = spk_id_dict
        self.f0_param_dict = f0_param_dict
        self.n_tokens = n_tokens
        self._pad_val = padding_value

        self.vals, self.fs, self.spk_ids, self.names = self._prepare_dataset(path, normalise_pitch)

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i], self.fs[i], self.spk_ids[i], self.names[i]

    def _prepare_dataset(self, path, normalise_pitch):
        fs, seqs, spk_ids, names = [], [], [], []
        with open(path, 'r') as f:
            for line in f.readlines():
                val_dict = eval(line)
                name = val_dict['audio'].split('_')[0]
                seqs.append(torch.IntTensor(val_dict['units']))
                if normalise_pitch:
                    f0 = torch.FloatTensor(val_dict['f0'])
                    ii = (f0 != 0)
                    f0[ii] -= self.f0_param_dict[name]['mean']
                    f0[ii] /= self.f0_param_dict[name]['std']
                    fs.append(f0)
                else:
                    fs.append((torch.FloatTensor(val_dict['f0'])))
                spk_ids.append(torch.IntTensor([self.spk_id_dict[name]]))
                names.append(val_dict['audio'])
        return pad_sequence(seqs, batch_first=True, padding_value=self.n_tokens), \
               pad_sequence(fs, batch_first=True, padding_value=self._pad_val), torch.concat(spk_ids).view(-1, 1), \
               names
