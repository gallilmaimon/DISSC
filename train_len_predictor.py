import argparse
import pickle
import os
import torch
from torch.utils.data import DataLoader

from dataset.len_dataset import LenDataset
from model.len_predictor import LenPredictor
from loss.len_loss import LenMSELoss, LenMAELoss, LenExactAccuracy, LenOneOffAccuracy, LenSmoothL1Loss, LenSumLoss
from utils import seed_everything, init_loggers, log_metrics


def train(data_path: str, device: str = 'cuda:0', args=None) -> None:
    _padding_value = -1

    out_path = args.out_path + '/len'
    train_logger, val_logger = init_loggers(out_path)

    with open(f'{args.data_path}/id_to_spkr.pkl', 'rb') as f:
        spk_id_dict = {v: k for (k, v) in dict(enumerate(pickle.load(f))).items()}

    ds_train = LenDataset(f'{data_path}/train.txt', spk_id_dict, args.n_tokens, _padding_value)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

    ds_val = LenDataset(f'{data_path}/val.txt', spk_id_dict, args.n_tokens, _padding_value)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    lens_train = ds_train.lens[ds_train.lens != _padding_value]
    model = LenPredictor(n_tokens=args.n_tokens, n_speakers=len(spk_id_dict), norm_mean=lens_train.mean(), norm_std=lens_train.std())
    model.to(device)

    torch.save((model.norm_mean, model.norm_std), out_path + '/len_norm_stats.pth')

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_s = LenSumLoss(pad_idx=_padding_value)
    mse = LenMSELoss(pad_idx=_padding_value)
    mae = LenMAELoss(pad_idx=_padding_value)
    acc = LenExactAccuracy(pad_idx=_padding_value)
    acc1 = LenOneOffAccuracy(pad_idx=_padding_value)

    best_mse = torch.inf

    for epoch in range(args.n_epochs):
        print(f'Epoch: {epoch}')
        model.train()
        total_train_loss = 0
        total_train_mse = 0
        total_train_mae = 0
        total_train_acc = 0
        total_train_acc1 = 0
        num_train_loss_samples = 0  # calculates the total number of samples which aren't padding in order to normalise loss

        for i, batch in enumerate(dl_train):
            seqs, lens, spk_id, _ = batch
            seqs = seqs.to(device)
            lens = lens.to(device)
            spk_id = spk_id.to(device)
            opt.zero_grad()

            preds = model(seqs, spk_id)
            loss = loss_s(preds, lens)
            loss.backward()
            opt.step()

            with torch.no_grad():
                total_train_loss += loss.detach()
                total_train_mse += mse(preds, lens)
                total_train_mae += mae(preds, lens)
                total_train_acc += acc(preds, lens)
                total_train_acc1 += acc1(preds, lens)
            cur_n_samples = (seqs != args.n_tokens).sum()
            num_train_loss_samples += cur_n_samples.detach().cpu()

            print(f'\r finished: {100 * i / len(dl_train):.2f}%, train loss: {loss / cur_n_samples:.5f}', end='')
        print()  # used to account for \r

        # validation
        model.eval()
        total_val_loss = 0
        total_val_mse = 0
        total_val_mae = 0
        total_val_acc = 0
        total_val_acc1 = 0
        num_val_loss_samples = 0  # calculates the total number of samples which aren't padding in order to normalise loss
        for i, batch in enumerate(dl_val):
            seqs, lens, spk_id, _ = batch
            seqs = seqs.to(device)
            lens = lens.to(device)
            spk_id = spk_id.to(device)
            with torch.no_grad():
                preds = model(seqs, spk_id)
                total_val_loss += loss_s(preds, lens)
                total_val_mse += mse(preds, lens)
                total_val_mae += mae(preds, lens)
                total_val_acc += acc(preds, lens)
                total_val_acc1 += acc1(preds, lens)
            num_val_loss_samples += (seqs != args.n_tokens).sum()

        # save best model
        if total_val_mse < best_mse:
            torch.save(model.state_dict(), out_path + '/best_model.pth')
            best_mse = total_val_mse

        log_metrics(train_logger, {"Loss": total_train_loss.detach().cpu() / num_train_loss_samples,
                                   "MSE": total_train_mse.detach().cpu() / num_train_loss_samples,
                                   "MAE": total_train_mae.detach().cpu() / num_train_loss_samples,
                                   "Accuracy": total_train_acc.detach().cpu() / num_train_loss_samples,
                                   "Accuracy_1": total_train_acc1.detach().cpu() / num_train_loss_samples}, epoch, 'train')
        log_metrics(val_logger, {"Loss": total_val_loss.cpu() / num_val_loss_samples.cpu(),
                                 "MSE": total_val_mse.cpu() / num_val_loss_samples.cpu(),
                                 "MAE": total_val_mae.cpu() / num_val_loss_samples.cpu(),
                                 "Accuracy": total_val_acc.cpu() / num_val_loss_samples.cpu(),
                                 "Accuracy_1": total_val_acc1.cpu() / num_val_loss_samples.cpu()}, epoch, 'val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='results/baseline', help='Path to save model and logs')
    parser.add_argument('--data_path', default='data/VCTK-corpus/hubert100', help='Path to sequence data')
    parser.add_argument('--n_tokens', default=100, type=int, help='number of unique HuBERT tokens to use (which represent how many clusters were used)')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', default=42, type=int, help='random seed, use -1 for non-determinism')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for train and inference')
    parser.add_argument('--learning_rate', default=3e-4, type=float, help='initial learning rate of the Adam optimiser')
    parser.add_argument('--n_epochs', default=100, type=int, help='number of training epochs')

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.out_path + '/len', exist_ok=True)
    train(args.data_path, args.device, args)