import pickle
import argparse
import torch
from torch.utils.data import DataLoader

from dataset.utils import get_spkrs_dict
from dataset.pitch_dataset import PitchDataset
from model.pitch_predictor import PitchPredictor
from loss.pitch_loss import PitchLoss, PitchRegLoss
from utils import seed_everything


def train(data_path: str, f0_path: str, device: str = 'cuda:0', args: argparse = None):
    with open(f0_path, 'rb') as f:
        f0_param_dict = pickle.load(f)
    spk_id_dict = get_spkrs_dict(f'{data_path}/train.txt')

    ds_train = PitchDataset(f'{data_path}/train.txt', spk_id_dict, f0_param_dict, nbins=args.n_bins)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

    ds_val = PitchDataset(f'{data_path}/val.txt', spk_id_dict, f0_param_dict, nbins=ds_train.nbins,
                          f_min=ds_train.f_min, scale=ds_train.scale)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=True)

    model = PitchPredictor(nbins=args.n_bins)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pitch_loss = PitchLoss(ds_train.f_min, ds_train.scale, ds_train.nbins)
    reg_loss = PitchRegLoss()

    for epoch in range(args.n_epochs):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Whether to train or inference in [\'train\']')
    parser.add_argument('--data_path', default='data/VCTK-corpus/hubert100', help='Path to sequence data')
    parser.add_argument('--f0_path', default='data/VCTK-corpus/hubert100/f0_stats.pkl', help='Pitch normalisation stats pickle')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', default=42, help='random seed, use -1 for non-determinism')
    parser.add_argument('--batch_size', default=32, help='batch size for train and inference')
    parser.add_argument('--learning_rate', default=3e-4, help='initial learning rate of the Adam optimiser')
    parser.add_argument('--n_epochs', default=100, help='number of training epochs')
    parser.add_argument('--n_bins', default=50, help='number of uniform bins for splitting the normalised frequencies')

    args = parser.parse_args()

    seed_everything(args.seed)
    train(args.data_path, args.f0_path, args.device, args)