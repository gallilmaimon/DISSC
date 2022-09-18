import argparse
import os
import random
import shutil

import torch
import pickle
import json
from torch.utils.data import DataLoader, Subset

from dataset.utils import get_spkrs_dict, prep_stats_tensors

# length prediction
from dataset.len_dataset import dedup_seq
from model.len_predictor import LenPredictor

# pitch prediction
from dataset.pitch_dataset import PitchDataset
from model.pitch_predictor import PitchPredictor

from utils import seed_everything, morph_seq_len


def _infer_sample(seqs, pitch, spk_id, name, f_min, scale, out_path, len_model=None, pitch_model=None) -> dict:
    in_seq = seqs[seqs != args.n_tokens].view(1, -1)
    if len_model:
        dd_seq, _ = dedup_seq(in_seq.cpu().numpy()[0])
        dd_seq = torch.tensor(dd_seq, device=in_seq.device).unsqueeze(0)
        with torch.no_grad():
            lens = len_model(dd_seq, spk_id)
            lens = torch.round(torch.clamp(lens[0], min=1)).int()
        out_seq = torch.repeat_interleave(dd_seq, lens).view(1, -1)
    else:
        out_seq = in_seq

    if pitch_model:
        with torch.no_grad():
            pitches = pitch_model.infer_freq(out_seq, spk_id, f_min, scale)
            pitches = pitches[0].cpu().numpy().tolist()
    else:  # If not predicting the pitch directly it is interpolated heuristically
        pitches = morph_seq_len(in_seq[0].cpu().numpy(), pitch.numpy(), lens.cpu().numpy()).tolist()
    out = {'units': out_seq[0].cpu().numpy().tolist(), 'f0': pitches, 'audio': name}
    with open(out_path, 'a+') as f:
        f.write(f'{json.dumps(out)}\n')
    return out

def infer(input_path: str, device: str = 'cuda:0', args=None) -> None:
    _padding_value = -1 if args.pred_len else -100

    spk_id_dict = get_spkrs_dict(f'{os.path.dirname(input_path)}/train.txt')
    out_path = f'{args.out_path}/{os.path.basename(input_path)}'

    with open(args.f0_path, 'rb') as f:
        f0_param_dict = pickle.load(f)
    id2pitch_mean, id2pitch_std = prep_stats_tensors(spk_id_dict, f0_param_dict)

    with open(args.f0_model + 'f0_params.pkl', 'rb') as f:
        n_bins, f_min, scale = pickle.load(f)

    ds = PitchDataset(input_path, spk_id_dict, f0_param_dict, n_bins=n_bins, f_min=f_min, scale=scale,
                      n_tokens=args.n_tokens, padding_value=_padding_value, normalise_pitch=False)
    dl = DataLoader(Subset(ds, range(args.n)), batch_size=1, shuffle=False)

    # define and load models
    len_model, pitch_model = None, None
    if args.pred_len:
        len_model = LenPredictor(n_tokens=args.n_tokens, n_speakers=len(spk_id_dict))
        len_model.to(device)
        len_model.eval()
        len_model.load_state_dict(torch.load(args.len_model))

    if args.pred_pitch:
        pitch_model = PitchPredictor(args.n_tokens, len(spk_id_dict), nbins=n_bins,
                                     id2pitch_mean=id2pitch_mean.to(args.device),
                                     id2pitch_std=id2pitch_std.to(args.device))
        pitch_model.to(device)
        pitch_model.eval()
        pitch_model.load_state_dict(torch.load(args.f0_model + 'best_model.pth'))

    # select target speakers if performing voice conversion
    target_spkrs = None
    if args.vc:
        if args.target_speakers:
            target_spkrs = args.target_speakers
        else:
            target_spkrs = random.sample(spk_id_dict.keys(), k=min(1, len(spk_id_dict.keys())))

    # remove existing files
    os.remove(out_path) if os.path.exists(out_path) else ''
    if target_spkrs:
        for t in target_spkrs:
            p = f'{args.out_path}/{t}_{os.path.basename(input_path)}'
            os.remove(p) if os.path.exists(p) else ''

    for i, batch in enumerate(dl):
        seqs, pitch, spk_id, name = batch
        seqs = seqs.to(device)
        spk_id = spk_id.to(device)
        pitch = pitch[0][pitch[0] != ds._pad_val]  # take only actual original pitch

        # reconstruction
        _infer_sample(seqs, pitch, spk_id, name[0], f_min, scale, out_path, len_model, pitch_model)

        # voice conversion
        if target_spkrs:
            for t in target_spkrs:
                spk_id[0][0] = spk_id_dict[t]
                _infer_sample(seqs, pitch, spk_id, name[0], f_min, scale, f'{args.out_path}/{t}_{os.path.basename(input_path)}', len_model, pitch_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='data/VCTK-corpus/hubert100/val.txt', help='Path to txt file of encoded HuBERT data')
    parser.add_argument('-n', default=10, type=int, help='number of samples to perform inference on')
    parser.add_argument('--out_path', default='data/VCTK-corpus/pred_hubert', help='Path to save predicted sequence')
    parser.add_argument('--pred_len', action='store_true', help='If true we predict the output length as well')
    parser.add_argument('--pred_pitch', action='store_true', help='If true we predict the output pitch as well')
    parser.add_argument('--len_model', default='results/baseline_new/len/best_model.pth', help='Path of len prediction model')
    parser.add_argument('--f0_model', default='results/baseline/pitch/', help='Path of pitch prediction model & stats')
    parser.add_argument('--n_tokens', default=100, type=int, help='number of unique HuBERT tokens to use (which represent how many clusters were used)')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', default=42, type=int, help='random seed, use -1 for non-determinism')
    parser.add_argument('--f0_path', default='data/VCTK-corpus/hubert100/f0_stats.pkl', help='Pitch normalisation stats pickle')
    parser.add_argument('--vc', action='store_true', help='If true we convert speakers and not only reconstruct')
    parser.add_argument('--target_speakers', nargs='+', default=None, help='Target speakers for VC. If none random speakers are used')

    args = parser.parse_args()

    assert args.pred_len | args.pred_pitch, "Inference must at least convert pitch or rhythm (or both)"

    seed_everything(args.seed)
    os.makedirs(args.out_path, exist_ok=True)
    os.remove(f'{args.out_path}/{os.path.basename(args.input_path)}') if os.path.exists(f'{args.out_path}/{os.path.basename(args.input_path)}') else ''
    infer(args.input_path, args.device, args)