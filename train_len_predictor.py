import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset.utils import get_spkrs_dict
from dataset.len_dataset import LenDataset
from model.len_predictor import LenPredictor
from loss.len_loss import LenLoss
from utils import seed_everything, init_loggers, log_metrics


def train(data_path: str, device: str = 'cuda:0', args=None) -> None:
    _padding_value = -1

    out_path = args.out_path + '/len'
    train_logger, val_logger = init_loggers(out_path)

    spk_id_dict = get_spkrs_dict(f'{data_path}/train.txt')

    ds_train = LenDataset(f'{data_path}/train.txt', spk_id_dict, args.n_tokens, _padding_value)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

    ds_val = LenDataset(f'{data_path}/val.txt', spk_id_dict, args.n_tokens, _padding_value)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    model = LenPredictor(n_tokens=args.n_tokens, n_speakers=len(spk_id_dict))
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    len_loss = LenLoss(pad_idx=_padding_value)

    best_loss = torch.inf

    for epoch in range(args.n_epochs):
        print(f'\nEpoch: {epoch}')

        model.train()
        total_train_loss = 0
        num_train_loss_samples = 0  # calculates the total number of samples which aren't padding in order to normalise loss

        for i, batch in enumerate(dl_train):
            seqs, lens, spk_id = batch
            seqs = seqs.to(device)
            lens = lens.to(device)
            spk_id = spk_id.to(device)
            opt.zero_grad()

            preds = model(seqs, spk_id)
            loss = len_loss(preds, lens)
            loss.backward()
            opt.step()

            total_train_loss += loss
            cur_n_samples = (seqs != _padding_value).sum()
            num_train_loss_samples += cur_n_samples.detach().cpu()

            print(f'\r finished: {100 * i / len(dl_train):.2f}%, train loss: {loss / cur_n_samples:.5f}', end='')
        print()  # used to account for \r

        # validation
        model.eval()
        total_val_loss = 0
        num_val_loss_samples = 0  # calculates the total number of samples which aren't padding in order to normalise loss
        for i, batch in enumerate(dl_val):
            seqs, lens, spk_id = batch
            seqs = seqs.to(device)
            lens = lens.to(device)
            spk_id = spk_id.to(device)
            with torch.no_grad():
                preds = model(seqs, spk_id)
                loss = len_loss(preds, lens)
            total_val_loss += loss
            num_val_loss_samples += (seqs != _padding_value).sum()

        # save best model
        if total_val_loss < best_loss:
            torch.save(model.state_dict(), out_path + '/best_model.pth')
            best_loss = total_val_loss

        log_metrics(train_logger, {"loss": total_train_loss.detach().cpu() / num_train_loss_samples}, epoch, 'train')
        log_metrics(val_logger, {"loss": total_val_loss.detach().cpu() / num_val_loss_samples.detach().cpu()}, epoch, 'val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Whether to train or inference in [\'train\']')
    parser.add_argument('--out_path', default='results/baseline', help='Path to save model and logs')
    parser.add_argument('--data_path', default='data/VCTK-corpus/hubert100', help='Path to sequence data')
    parser.add_argument('--n_tokens', default=100, help='number of unique HuBERT tokens to use (which represent how many clusters were used)')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', default=42, help='random seed, use -1 for non-determinism')
    parser.add_argument('--batch_size', default=32, help='batch size for train and inference')
    parser.add_argument('--learning_rate', default=3e-4, help='initial learning rate of the Adam optimiser')
    parser.add_argument('--n_epochs', default=200, help='number of training epochs')
    parser.add_argument('--n_bins', default=50, help='number of uniform bins for splitting the normalised frequencies')

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.out_path + '/len', exist_ok=True)
    train(args.data_path, args.device, args)