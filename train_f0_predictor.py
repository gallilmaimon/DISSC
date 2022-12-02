import os
import pickle
import argparse
import torch
from torch.utils.data import DataLoader

from dataset.utils import prep_stats_tensors
from dataset.pitch_dataset import PitchDataset
from model.pitch_predictor import PitchPredictor
from loss.pitch_loss import PitchLoss, PitchMAE, PitchMSE
from utils import seed_everything, init_loggers, log_metrics


def train(data_path: str, f0_path: str, device: str = 'cuda:0', args: argparse = None):
    _padding_value = -100  # this needs to be not likely to be encountered as label (which is whitened), and not as prob (0-1)

    out_path = args.out_path + '/pitch'
    train_logger, val_logger = init_loggers(out_path)

    with open(f0_path, 'rb') as f:
        f0_param_dict = pickle.load(f)
    with open(f'{args.data_path}/id_to_spkr.pkl', 'rb') as f:
        spk_id_dict = {v: k for (k, v) in dict(enumerate(pickle.load(f))).items()}
    id2pitch_mean, id2pitch_std = prep_stats_tensors(spk_id_dict, f0_param_dict)

    ds_train = PitchDataset(f'{data_path}/train.txt', spk_id_dict, f0_param_dict, n_tokens=args.n_tokens,
                            padding_value=_padding_value)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    ds_val = PitchDataset(f'{data_path}/val.txt', spk_id_dict, f0_param_dict, n_tokens=args.n_tokens, 
                          padding_value=_padding_value)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    model = PitchPredictor(args.n_tokens, len(spk_id_dict), id2pitch_mean=id2pitch_mean.to(args.device),
                           id2pitch_std=id2pitch_std.to(args.device))
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pitch_loss = PitchLoss(id2pitch_mean.to(args.device), id2pitch_std.to(args.device), pad_idx=_padding_value)
    mae = PitchMAE(id2pitch_mean.to(args.device), id2pitch_std.to(args.device), pad_idx=_padding_value)
    mse = PitchMSE(id2pitch_mean.to(args.device), id2pitch_std.to(args.device), pad_idx=_padding_value)

    best_mae = torch.inf
    for epoch in range(args.n_epochs):
        print(f'\nEpoch: {epoch}')
        model.train()
        total_train_loss = 0
        total_train_mse = 0
        total_train_mae = 0
        num_train_loss_samples = 0  # calculates the total number of samples which aren't padding in order to normalise loss
        for i, batch in enumerate(dl_train):
            seqs, gts_reg, spk_id, _ = batch
            seqs = seqs.to(device)
            gts_reg = gts_reg.to(device)
            spk_id = spk_id.to(device)
            opt.zero_grad()

            cls_preds, reg_preds = model(seqs, spk_id)
            loss = pitch_loss(cls_preds, reg_preds, gts_reg, spk_id)
            loss.backward()
            opt.step()
            total_train_loss += loss
            cur_n_samples = (gts_reg != _padding_value).sum()
            num_train_loss_samples += cur_n_samples

            with torch.no_grad():
                freqs = model.calc_freq(cls_preds, reg_preds, spk_id)
                total_train_mae += mae(freqs, gts_reg, spk_id)
                total_train_mse += mse(freqs, gts_reg, spk_id)
            print(f'\r finished: {100 * i / len(dl_train):.2f}%, train loss: '
                  f'{loss / cur_n_samples.detach().cpu():.5f}', end='')
        print()  # used to account for \r

        # validation
        model.eval()
        total_val_loss = 0
        total_val_mse = 0
        total_val_mae = 0
        num_val_loss_samples = 0  # calculates the total number of samples which aren't padding in order to normalise loss
        for i, batch in enumerate(dl_val):
            seqs, gts_reg, spk_id, _ = batch
            seqs = seqs.to(device)
            gts_reg = gts_reg.to(device)
            spk_id = spk_id.to(device)

            num_val_loss_samples += (gts_reg != _padding_value).sum()
            with torch.no_grad():
                cls_preds, reg_preds = model(seqs, spk_id)
                total_val_loss += pitch_loss(cls_preds, reg_preds, gts_reg, spk_id)
                freqs = model.calc_freq(cls_preds, reg_preds, spk_id)
                total_val_mae += mae(freqs, gts_reg, spk_id)
                total_val_mse += mse(freqs, gts_reg, spk_id)

        # save best model
        if total_val_mae < best_mae:
            torch.save(model.state_dict(), out_path + '/best_model.pth')
            best_mae = total_val_mae

        log_metrics(train_logger, {"loss": total_train_loss.detach().cpu() / num_train_loss_samples.detach().cpu(),
                                   'MSE': total_train_mse.detach().cpu() / num_train_loss_samples.detach().cpu(),
                                   'MAE': total_train_mae.detach().cpu() / num_train_loss_samples.detach().cpu()}, epoch, 'train')
        log_metrics(val_logger, {'loss': total_val_loss.detach().cpu() / num_val_loss_samples.detach().cpu(),
                                 'MSE': total_val_mse.detach().cpu() / num_val_loss_samples.detach().cpu(),
                                 'MAE': total_val_mae.detach().cpu() / num_val_loss_samples.detach().cpu()}, epoch, 'val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='results/baseline', help='Path to save model and logs')
    parser.add_argument('--data_path', default='data/ESD/hubert100', help='Path to sequence data')
    parser.add_argument('--n_tokens', default=100, type=int, help='number of unique HuBERT tokens to use (which represent how many clusters were used)')
    parser.add_argument('--f0_path', default='data/ESD/hubert100/f0_stats.pkl', help='Pitch normalisation stats pickle')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', default=42, type=int, help='random seed, use -1 for non-determinism')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for train and inference')
    parser.add_argument('--learning_rate', default=3e-4, type=float, help='initial learning rate of the Adam optimiser')
    parser.add_argument('--n_epochs', default=20, type=int, help='number of training epochs')

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.out_path + '/pitch', exist_ok=True)
    train(args.data_path, args.f0_path, args.device, args)
