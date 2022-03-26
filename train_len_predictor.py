import torch
from torch.utils.data import DataLoader

from dataset.utils import get_spkrs_dict
from dataset.len_dataset import LenDataset
from model.len_predictor import LenPredictor
from loss.len_loss import LenLoss


def train(data_path: str, device: str = 'cuda:0') -> None:
    spk_id_dict = get_spkrs_dict(f'{data_path}/train.txt')

    ds_train = LenDataset(f'{data_path}/train.txt', spk_id_dict)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)

    ds_val = LenDataset(f'{data_path}/val.txt', spk_id_dict)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)

    model = LenPredictor()
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    len_loss = LenLoss()

    for epoch in range(100):
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
    data_path = 'data/VCTK-corpus/hubert100'
    device = 'cuda:0'
    train(data_path, device)