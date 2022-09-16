import torch
from torch import nn

class LenMSELoss(nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenMSELoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, lens):
        mask = (lens != self.pad_idx)
        total_loss = self.mse(preds, lens)
        return (mask * total_loss).sum()

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