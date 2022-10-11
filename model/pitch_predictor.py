import torch
from torch import nn

class PitchPredictor(nn.Module):
    def __init__(self, n_tokens=100, n_speakers=199, emb_size=32, id2pitch_mean=None, id2pitch_std=None):
        super(PitchPredictor, self).__init__()

        # used for returning actual pitch, not just normalised results
        self.id2pitch_mean = id2pitch_mean
        self.id2pitch_std = id2pitch_std

        self.token_emb = nn.Embedding(n_tokens + 1, emb_size, padding_idx=n_tokens)
        self.spk_emb = nn.Embedding(n_speakers + 1, emb_size, padding_idx=n_speakers)
        self.leaky = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.0)

        self.cnn1 = nn.Conv1d(2 * emb_size, 128, kernel_size=(3,), padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.cnn11 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn11 = nn.BatchNorm1d(128)
        self.cnn12 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn12 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)

        self.cnn_class = nn.Conv1d(128, 1, kernel_size=(1,), padding=0)
        self.cnn_reg = nn.Conv1d(128, 1, kernel_size=(1,), padding=0)

    def forward(self, seq, spk_id):
        emb_seq = self.token_emb(seq)
        emb_spk = torch.repeat_interleave(self.spk_emb(spk_id), seq.shape[-1], dim=1)
        emb_seq = torch.cat([emb_seq, emb_spk], dim=-1)

        cnn1 = self.leaky(self.dropout(self.bn1(self.cnn1(emb_seq.transpose(1, 2)))))
        cnn1 = self.leaky(self.dropout(self.bn11(self.cnn11(cnn1))))
        cnn1 = self.leaky(self.dropout(self.bn12(self.cnn12(cnn1))))

        cnn2 = self.leaky(self.dropout(self.cnn2(cnn1)))

        return self.cnn_class(cnn2).squeeze(1), self.cnn_reg(cnn2).squeeze(1)

    def infer_freq(self, seq, spk_id, norm=False):
        class_preds, reg_preds = self(seq, spk_id)
        return self.calc_freq(class_preds, reg_preds, spk_id, norm)

    def calc_freq(self, class_preds, reg_preds, spk_id, norm=False):
        spk_mask = (class_preds > 0)
        if not norm:
            reg_preds = self.id2pitch_mean[spk_id.long()] + reg_preds * self.id2pitch_std[spk_id.long()]
        return spk_mask * reg_preds
