import torch
from torch import nn


class LenPredictor(nn.Module):
    def __init__(self, n_tokens=100, n_speakers=99, emb_size=32, masking_rate=0.2, norm_mean=torch.tensor(0),
                 norm_std=torch.tensor(1)):
        super(LenPredictor, self).__init__()
        self.keep_rate = 1 - masking_rate

        # used to effectively normalise the labels
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.token_emb = nn.Embedding(n_tokens + 1, emb_size, padding_idx=n_tokens)
        self.spk_emb = nn.Embedding(n_speakers, emb_size)
        self.leaky = nn.LeakyReLU()
        self.cnn1 = nn.Conv1d(2 * emb_size, 128, kernel_size=(3,), padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.cnn11 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn11 = nn.BatchNorm1d(128)
        self.cnn12 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn12 = nn.BatchNorm1d(128)
        self.cnn13 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn13 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn14 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn15 = nn.BatchNorm1d(128)
        self.cnn16 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn16 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(128, 1, kernel_size=(3,), padding=1)

    def forward(self, seq, spk_id):
        emb_seq = self.token_emb(seq)
        if self.training:  # This masks part of the sequence to avoid overfit and encourage disentanglement
            mask = torch.cuda.FloatTensor(emb_seq.shape[0], emb_seq.shape[1]).uniform_() > self.keep_rate
            emb_seq[mask] = 0

        emb_spk = torch.repeat_interleave(self.spk_emb(spk_id), seq.shape[-1], dim=1)
        emb_seq = torch.cat([emb_seq, emb_spk], dim=-1)

        cnn1 = self.leaky(self.bn1(self.cnn1(emb_seq.transpose(1, 2))))
        cnn1 = self.leaky(self.bn11(self.cnn11(cnn1)))
        cnn1 = self.leaky(self.bn12(self.cnn12(cnn1)))
        cnn1 = self.leaky(self.bn13(self.cnn13(cnn1)))
        cnn1 = self.leaky(self.bn14(self.cnn14(cnn1)))
        cnn1 = self.leaky(self.bn15(self.cnn15(cnn1)))
        cnn1 = self.leaky(self.bn16(self.cnn16(cnn1)))

        return self.cnn2(cnn1).squeeze(1) * self.norm_std + self.norm_mean
