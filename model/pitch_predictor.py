import torch
from torch import nn

class PitchPredictor(nn.Module):
    def __init__(self, n_tokens=100, n_speakers=199, emb_size=32, nbins=50, id2pitch_mean=None, id2pitch_std=None):
        super(PitchPredictor, self).__init__()
        self.nbins = nbins

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

        self.cnn2 = nn.Conv1d(128, nbins, kernel_size=(3,), padding=1)

    def forward(self, seq, spk_id):
        emb_seq = self.token_emb(seq)
        emb_spk = torch.repeat_interleave(self.spk_emb(spk_id), seq.shape[-1], dim=1)
        emb_seq = torch.cat([emb_seq, emb_spk], dim=-1)

        cnn1 = self.leaky(self.dropout(self.bn1(self.cnn1(emb_seq.transpose(1, 2)))))
        cnn1 = self.leaky(self.dropout(self.bn11(self.cnn11(cnn1))))
        cnn1 = self.leaky(self.dropout(self.bn12(self.cnn12(cnn1))))

        return self.cnn2(cnn1).squeeze(1)

    def infer_norm_freq(self, seq, spk_id, fmin, scale):
        return self.calc_norm_freq(self(seq, spk_id), fmin, scale)

    def calc_norm_freq(self, preds, fmin, scale):
        preds = torch.sigmoid(preds.transpose(1, 2))  # calculate class probs
        # Uses middle value for bin representative
        f_weights = torch.linspace(fmin + 0.5 * scale, fmin + (self.nbins - 0.5) * scale, self.nbins, device=preds.device)
        return torch.inner(preds, f_weights)

    def infer_freq(self, seq, spk_id, fmin, scale):
        norm_pitch = self.infer_norm_freq(seq, spk_id, fmin, scale)
        return self.id2pitch_mean[spk_id.long()] + (norm_pitch * self.id2pitch_std[spk_id.long()])

    def calc_freq(self, preds, fmin, scale, spk_id):
        norm_pitch = self.calc_norm_freq(preds, fmin, scale)
        return self.id2pitch_mean[spk_id.long()] + (norm_pitch * self.id2pitch_std[spk_id.long()])
