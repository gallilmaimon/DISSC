from torch import nn

class LenLoss(nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenLoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, lens):
        mask = (lens != self.pad_idx)
        total_loss = self.mse(preds, lens)
        return (mask * total_loss).sum()