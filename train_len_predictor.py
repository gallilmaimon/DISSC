import argparse
import torch
from torch.utils.data import DataLoader

from dataset.utils import get_spkrs_dict
from dataset.len_dataset import LenDataset
from model.len_predictor import LenPredictor
from loss.len_loss import LenLoss
from utils import seed_everything


def train(data_path: str, device: str = 'cuda:0') -> None:
    spk_id_dict = get_spkrs_dict(f'{data_path}/train.txt')

    ds_train = LenDataset(f'{data_path}/train.txt', spk_id_dict)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

    ds_val = LenDataset(f'{data_path}/val.txt', spk_id_dict)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    model = LenPredictor()
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    len_loss = LenLoss()

    for epoch in range(args.n_epochs):
        print(f'\nEpoch: {epoch}')

        model.train()
        total_train_loss = 0
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

            print(f'\r finished: {100 * i / len(dl_train):.2f}%, train loss: {loss:.5f}', end='')

        # validation
        model.eval()
        total_val_loss = 0
        for i, batch in enumerate(dl_val):
            seqs, lens, spk_id = batch
            seqs = seqs.to(device)
            lens = lens.to(device)
            spk_id = spk_id.to(device)
            with torch.no_grad():
                preds = model(seqs, spk_id)
                loss = len_loss(preds, lens)
            total_val_loss += loss

        print(f'\ntotal_train_loss: {total_train_loss / len(dl_train):.5f}')
        print(f'total_val_loss: {total_val_loss / len(dl_val):.5f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Whether to train or inference in [\'train\']')
    parser.add_argument('--data_path', default='data/VCTK-corpus/hubert100', help='Path to sequence data')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', default=42, help='random seed, use -1 for non-determinism')
    parser.add_argument('--batch_size', default=32, help='batch size for train and inference')
    parser.add_argument('--learning_rate', default=3e-4, help='initial learning rate of the Adam optimiser')
    parser.add_argument('--n_epochs', default=100, help='number of training epochs')
    parser.add_argument('--n_bins', default=50, help='number of uniform bins for splitting the normalised frequencies')

    args = parser.parse_args()

    seed_everything(args.seed)
    train(args.data_path, args.device)