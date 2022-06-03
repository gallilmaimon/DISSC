import torch
from torch import nn
from torchvision.transforms import GaussianBlur

# constants
EPS = 0.001

class PitchLoss(nn.Module):
    def __init__(self, f_min, scale, nbins, pad_idx=-1):
        super(PitchLoss, self).__init__()
        self.pad_idx = pad_idx
        self.f_min = f_min
        self.scale = scale
        self.nbins = nbins
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, gt):
        # transforming labels to blurred one-hot is done in loss for memory efficiency reasons
        gt, _, _ = prepare_f0(gt, self.nbins, self.f_min, self.scale, self.pad_idx)
        mask = (gt != self.pad_idx)
        total_loss = self.bce(preds, gt)
        return (mask * total_loss).sum()


class NormPitchRegLoss(nn.Module):
    def __init__(self, pad_idx=100):
        super(NormPitchRegLoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, gts):
        mask = (gts != self.pad_idx)
        total_loss = self.mse(preds, gts)
        return (mask * total_loss).sum()


class PitchRegLoss(nn.Module):
    def __init__(self, id2std=None, pad_idx=100):
        super(PitchRegLoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.MSELoss(reduction='none')
        self.id2std = id2std

    def forward(self, preds, gts, spk_ids):
        mask = (gts != self.pad_idx)
        total_loss = self.mse(preds, gts)
        pitch_weight = self.id2std[spk_ids.long()]
        return (mask * total_loss * pitch_weight).sum()


def quantise_f0(fs, nbins=50, f_min=None, scale=None):
    if f_min is None:
        f_min = fs.min()
    if scale is None:
        scale = (fs.max() + EPS - f_min) / nbins
    q_fs = torch.clip(torch.div(fs - f_min, scale, rounding_mode='floor').long(), min=0, max=nbins - 1)
    return nn.functional.one_hot(q_fs, num_classes=nbins), f_min, scale


def prepare_f0(fs, nbins=50, f_min=None, scale=None, pad_idx=-100):
    res = torch.zeros((fs.shape[0], fs.shape[1], nbins)).long().to(fs.device)
    res[fs != pad_idx], fmin, scale = quantise_f0(fs[fs != pad_idx], nbins, f_min, scale)

    filt = GaussianBlur(kernel_size=(5, 1), sigma=0.5)
    res = filt(res.float())
    res[fs == pad_idx] = pad_idx
    return res, fmin, scale