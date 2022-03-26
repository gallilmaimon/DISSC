import torch
from torch.utils.data import DataLoader

from dataset.utils import get_spkrs_dict
from dataset.pitch_dataset import PitchDataset
from model.pitch_predictor import PitchPredictor
from loss.pitch_loss import PitchLoss, PitchRegLoss

def train(data_path: str, f0_path: str, device: str = 'cuda:0'):
    f0_param_dict = torch.load(f0_path)
    spk_id_dict = get_spkrs_dict(f'{data_path}/train.txt')

    ds_train = PitchDataset(f'{data_path}/train.txt', spk_id_dict, f0_param_dict, nbins=50)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)

    ds_val = PitchDataset(f'{data_path}/val.txt', spk_id_dict, f0_param_dict, nbins=ds_train.nbins,
                          f_min=ds_train.f_min, scale=ds_train.scale)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=True)

    model = PitchPredictor(nbins=50)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    pitch_loss = PitchLoss(ds_train.f_min, ds_train.scale, ds_train.nbins)
    reg_loss = PitchRegLoss()

    for epoch in range(100):
        print(f'\nEpoch: {epoch}')

        model.train()
        total_train_loss = 0
        total_train_mse = 0
        for i, batch in enumerate(dl_train):
            seqs, gts_reg, spk_id = batch
            seqs = seqs.to(device)
            gts_reg = gts_reg.to(device)
            spk_id = spk_id.to(device)
            opt.zero_grad()

            preds = model(seqs, spk_id)
            loss = pitch_loss(preds.transpose(1, 2), gts_reg)
            loss.backward()
            opt.step()
            total_train_loss += loss
            total_train_mse += reg_loss(model.calc_norm_freq(preds, ds_train.f_min, ds_train.scale), gts_reg)

            print(f'\r finished: {100 * i / len(dl_train):.2f}%, train loss: {loss:.5f}', end='')

        # validation
        model.eval()
        total_val_loss = 0
        total_val_mse = 0
        for i, batch in enumerate(dl_val):
            seqs, gts_reg, spk_id = batch
            seqs = seqs.to(device)
            gts_reg = gts_reg.to(device)
            spk_id = spk_id.to(device)
            with torch.no_grad():
                preds = model(seqs, spk_id)
                loss = pitch_loss(preds.transpose(1, 2), gts_reg)
            total_val_loss += loss
            total_val_mse += reg_loss(model.calc_norm_freq(preds, ds_val.f_min, ds_val.scale), gts_reg)

        print(
            f'\ntotal_train_loss: {total_train_loss / len(dl_train):.5f}, train MSE: {total_train_mse / len(dl_train):.5f}')
        print(f'total_val_loss: {total_val_loss / len(dl_val):.5f}, val MSE: {total_val_mse / len(dl_val):.5f}')


if __name__ == '__main__':
    data_path = 'data/VCTK-corpus/hubert100'
    f0_path = 'data/VCTK-corpus/hubert100/f0_stats.th'
    device = 'cuda:0'
    train(data_path, f0_path, device)