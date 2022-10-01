from torch import nn

# constants
EPS = 0.001

class PitchLoss(nn.Module):
    def __init__(self, id2mean=None, id2std=None, pad_idx=-1):
        super(PitchLoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.L1Loss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.id2std = id2std
        self.id2mean = id2mean

    def forward(self, class_preds, reg_preds, gts, spk_ids):
        mask = (gts != self.pad_idx)

        spk_labels = (gts != 0)
        loss1 = (mask * self.bce(class_preds, spk_labels.float().detach())).sum()

        std = self.id2std[spk_ids.long()]
        mean = self.id2mean[spk_ids.long()]
        preds = mean + std * reg_preds
        gts = mean + std * gts
        total_loss = self.mse(preds, gts)
        loss2 = (mask * total_loss * spk_labels).sum()
        return 100 * loss1 + 1 * loss2


class PitchMAE(nn.Module):
    def __init__(self, id2mean=None, id2std=None, pad_idx=-1):
        super(PitchMAE, self).__init__()
        self.pad_idx = pad_idx
        self.mae = nn.L1Loss(reduction='none')
        self.id2std = id2std
        self.id2mean = id2mean

    def forward(self, preds, gts, spk_ids):
        std = self.id2std[spk_ids.long()]
        mean = self.id2mean[spk_ids.long()]
        mask = (gts != self.pad_idx)
        ii = (gts != 0)
        gts = (mean + std * gts) * ii
        total_loss = self.mae(preds, gts)
        return (mask * total_loss).sum()

class PitchMSE(nn.Module):
    def __init__(self, id2mean=None, id2std=None, pad_idx=-1):
        super(PitchMSE, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.MSELoss(reduction='none')
        self.id2std = id2std
        self.id2mean = id2mean

    def forward(self, preds, gts, spk_ids):
        std = self.id2std[spk_ids.long()]
        mean = self.id2mean[spk_ids.long()]
        mask = (gts != self.pad_idx)
        ii = (gts != 0)
        gts = (mean + std * gts) * ii
        total_loss = self.mse(preds, gts)
        return (mask * total_loss).sum()