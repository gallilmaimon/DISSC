import torch
from torch import nn
import torch.nn.functional as F

class LenMSELoss(nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenMSELoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, lens):
        mask = (lens != self.pad_idx)
        total_loss = self.mse(preds, lens)
        return (mask * total_loss).sum()

class LenSumLoss(nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenSumLoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, lens):
        # This is used to encourage nearby errors to cancel out, as to not cause a length bias
        diff4 = (F.avg_pool2d((preds - lens).unsqueeze(0), (1, 4)) * 4) ** 2
        diff_mask4 = ~F.max_pool2d((lens == self.pad_idx).unsqueeze(0).float(), (1, 4)).bool()
        diff_loss4 = (diff_mask4 * diff4).sum()

        mask = (lens != self.pad_idx)
        total_loss = self.mse(preds, lens)
        return (mask * total_loss).sum()  + 0.5 * diff_loss4


class LenMAELoss(nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenMAELoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.L1Loss(reduction='none')

    def forward(self, preds, lens):
        mask = (lens != self.pad_idx)
        total_loss = self.mse(preds, lens)
        return (mask * total_loss).sum()

class LenSmoothL1Loss(nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenSmoothL1Loss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.SmoothL1Loss(reduction='none')

    def forward(self, preds, lens):
        mask = (lens != self.pad_idx)
        total_loss = self.mse(preds, lens)
        return (mask * total_loss).sum()

class LenExactAccuracy(nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenExactAccuracy, self).__init__()
        self.pad_idx = pad_idx

    def forward(self, preds, lens):
        mask = (lens != self.pad_idx)
        preds = torch.round(torch.clamp(preds, min=1)).int()
        total_accuracy = (preds==lens)
        return (mask * total_accuracy).sum()

class LenOneOffAccuracy(nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenOneOffAccuracy, self).__init__()
        self.pad_idx = pad_idx

    def forward(self, preds, lens):
        mask = (lens != self.pad_idx)
        preds = torch.round(torch.clamp(preds, min=1)).int()
        total_accuracy = ((preds-lens).abs() <= 1)
        return (mask * total_accuracy).sum()