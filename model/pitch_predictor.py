import torch
from torch import nn

class PitchPredictor(nn.Module):
    def __init__(self, n_tokens=100, n_speakers=199, emb_size=32, masking_rate=0.4, id2pitch_mean=None,
                 id2pitch_std=None):
        super(PitchPredictor, self).__init__()
        self.keep_rate = 1 - masking_rate

        # used for returning actual pitch, not just normalised results
        self.id2pitch_mean = id2pitch_mean
        self.id2pitch_std = id2pitch_std

        self.token_emb = nn.Embedding(n_tokens + 1, emb_size, padding_idx=n_tokens)
        self.spk_emb = nn.Embedding(n_speakers + 1, emb_size, padding_idx=n_speakers)
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
        self.cnn17 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn17 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.cnn_class1 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn_c1 = nn.BatchNorm1d(128)
        self.cnn_class2 = nn.Conv1d(128, 1, kernel_size=(1,), padding=0)
        self.cnn_reg1 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.bn_r1 = nn.BatchNorm1d(128)
        self.cnn_reg2 = nn.Conv1d(128, 1, kernel_size=(1,), padding=0)

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
        cnn1 = self.leaky(self.bn17(self.cnn17(cnn1)))

        cnn2 = self.leaky(self.cnn2(cnn1))
        cnn_class = self.leaky(self.bn_c1(self.cnn_class1(cnn2)))
        cnn_reg = self.leaky(self.bn_r1(self.cnn_reg1(cnn2)))
        return self.cnn_class2(cnn_class).squeeze(1), self.cnn_reg2(cnn_reg).squeeze(1)

    def infer_freq(self, seq, spk_id, norm=False):
        class_preds, reg_preds = self(seq, spk_id)
        return self.calc_freq(class_preds, reg_preds, spk_id, norm)

    def calc_freq(self, class_preds, reg_preds, spk_id, norm=False):
        spk_mask = (class_preds > 0)
        if not norm:
            reg_preds = self.id2pitch_mean[spk_id.long()] + reg_preds * self.id2pitch_std[spk_id.long()]
        return spk_mask * reg_preds
