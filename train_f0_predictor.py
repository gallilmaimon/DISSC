import os
import pickle
import argparse
import torch
from torch.utils.data import DataLoader

from dataset.utils import get_spkrs_dict, prep_stats_tensors
from dataset.pitch_dataset import PitchDataset
from model.pitch_predictor import PitchPredictor
from loss.pitch_loss import PitchLoss, PitchRegLoss
from utils import seed_everything, init_loggers, log_metrics


def train(data_path: str, f0_path: str, device: str = 'cuda:0', args: argparse = None):
    _padding_value = -100  # this needs to be not likely to be encountered as label (which is whitened), and not as prob (0-1)

    out_path = args.out_path + '/pitch'
    train_logger, val_logger = init_loggers(out_path)

    with open(f0_path, 'rb') as f:
        f0_param_dict = pickle.load(f)
    spk_id_dict = get_spkrs_dict(f'{data_path}/train.txt')

    id2pitch_mean, id2pitch_std = prep_stats_tensors(spk_id_dict, f0_param_dict)

    ds_train = PitchDataset(f'{data_path}/train.txt', spk_id_dict, f0_param_dict, n_bins=args.n_bins,
                            n_tokens=args.n_tokens, padding_value=_padding_value)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

    ds_val = PitchDataset(f'{data_path}/val.txt', spk_id_dict, f0_param_dict, n_bins=ds_train.n_bins,
                          f_min=ds_train.f_min, scale=ds_train.scale, n_tokens=args.n_tokens,
                          padding_value=_padding_value)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    with open(out_path + '/f0_params.pkl', 'wb') as f:
        pickle.dump([ds_train.n_bins, ds_train.f_min, ds_train.scale], f)

    model = PitchPredictor(args.n_tokens, len(spk_id_dict), nbins=args.n_bins,
                           id2pitch_mean=id2pitch_mean.to(args.device), id2pitch_std=id2pitch_std.to(args.device))

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pitch_loss = PitchLoss(ds_train.f_min, ds_train.scale, ds_train.n_bins, pad_idx=_padding_value)
    reg_metric = PitchRegLoss(id2pitch_std.to(args.device), pad_idx=_padding_value)

    best_mse = torch.inf
    for epoch in range(args.n_epochs):
        print(f'\nEpoch: {epoch}')
        model.train()
        total_train_loss = 0
        total_train_mse = 0
        num_train_loss_samples = 0  # calculates the total number of samples which aren't padding in order to normalise loss
        for i, batch in enumerate(dl_train):
            seqs, gts_reg, spk_id, _ = batch
            seqs = seqs.to(device)
            gts_reg = gts_reg.to(device)
            spk_id = spk_id.to(device)
            opt.zero_grad()

            preds = model(seqs, spk_id)
            loss = pitch_loss(preds.transpose(1, 2), gts_reg)
            loss.backward()
            opt.step()
            total_train_loss += loss
            cur_n_samples = (seqs != _padding_value).sum()
            num_train_loss_samples += cur_n_samples

            with torch.no_grad():
                total_train_mse += reg_metric(model.calc_norm_freq(preds, ds_train.f_min, ds_train.scale), gts_reg,
                                              spk_id)
            print(f'\r finished: {100 * i / len(dl_train):.2f}%, train loss: '
                  f'{loss / cur_n_samples.detach().cpu() / ds_train.n_bins:.5f}', end='')
        print()  # used to account for \r

        # validation
        model.eval()
        total_val_loss = 0
        total_val_mse = 0
        num_val_loss_samples = 0  # calculates the total number of samples which aren't padding in order to normalise loss
        for i, batch in enumerate(dl_val):
            seqs, gts_reg, spk_id, _ = batch
            seqs = seqs.to(device)
            gts_reg = gts_reg.to(device)
            spk_id = spk_id.to(device)

            num_val_loss_samples += (seqs != _padding_value).sum()
            with torch.no_grad():
                preds = model(seqs, spk_id)
                total_val_loss += pitch_loss(preds.transpose(1, 2), gts_reg)
                total_val_mse += reg_metric(model.calc_norm_freq(preds, ds_val.f_min, ds_val.scale), gts_reg, spk_id)

        # save best model
        if total_val_mse < best_mse:
            torch.save(model.state_dict(), out_path + '/best_model.pth')
            best_mse = total_val_mse

        log_metrics(train_logger, {"loss": total_train_loss.detach().cpu() / num_train_loss_samples.detach().cpu() / ds_train.n_bins,
                                   'MSE': total_train_mse.detach().cpu() / num_train_loss_samples.detach().cpu()}, epoch, 'train')
        log_metrics(val_logger, {'loss': total_val_loss.detach().cpu() / num_val_loss_samples.detach().cpu() / ds_val.n_bins,
                                 'MSE': total_val_mse.detach().cpu() / num_val_loss_samples.detach().cpu()}, epoch, 'val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Whether to train or inference in [\'train\']')
    parser.add_argument('--out_path', default='results/baseline', help='Path to save model and logs')
    parser.add_argument('--data_path', default='data/VCTK-corpus/hubert100', help='Path to sequence data')
    parser.add_argument('--n_tokens', default=100, help='number of unique HuBERT tokens to use (which represent how many clusters were used)')
    parser.add_argument('--f0_path', default='data/VCTK-corpus/hubert100/f0_stats.pkl', help='Pitch normalisation stats pickle')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', default=42, help='random seed, use -1 for non-determinism')
    parser.add_argument('--batch_size', default=32, help='batch size for train and inference')
    parser.add_argument('--learning_rate', default=3e-4, help='initial learning rate of the Adam optimiser')
    parser.add_argument('--n_epochs', default=200, help='number of training epochs')
    parser.add_argument('--n_bins', default=50, help='number of uniform bins for splitting the normalised frequencies')

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.out_path + '/pitch', exist_ok=True)
    train(args.data_path, args.f0_path, args.device, args)
