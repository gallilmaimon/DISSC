import os
import argparse
import pandas as pd

import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.utils.metric_stats import EER


def prep_sample(ver_model, path):
    signal, sr = torchaudio.load(str(path), channels_first=False)
    return ver_model.audio_normalizer(signal, sr)

def verify_files(ver_model, path_x, path_y):

    waveform_x = prep_sample(ver_model, path_x)
    waveform_y = prep_sample(ver_model, path_y)
    # Fake batches:
    batch_x = waveform_x.unsqueeze(0)
    batch_y = waveform_y.unsqueeze(0)
    # Verify:
    score, decision = ver_model.verify_batch(batch_x, batch_y)
    # Squeeze:
    return score[0], decision[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='../results/vctk/', help='base path to data and sample CSV')
    parser.add_argument('--gt_path', default='/cs/dataset/Download/adiyoss/vctk/wav16_trimmed_padded/', help='path to real original data')
    parser.add_argument('--file_suffix', default='_mic2.flac', help='suffix to gt audio files')
    parser.add_argument('--method', default='sr', help='conversion type, as in file name')
    parser.add_argument('--device', default='cuda:0', help='Torch device')
    args = parser.parse_args()

    sample_csv = f'{args.base_path}/speaker_verification.csv'
    syn_path = f'{args.base_path}/sv/{args.method}/'

    df = pd.read_csv(sample_csv, index_col=0)
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir="../pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device":args.device})
    verification.to(args.device)
    verification.eval()

    scores = {0: [], 1: []}
    for _, row in df.iterrows():
        print(f'\r{_}', end='')
        gt = f'{args.gt_path}/{row.ref}{args.file_suffix}'
        syn = f'{syn_path}/{row.syn_trgt}/{row.syn_sample}.wav'
        if os.path.isfile(gt) and os.path.isfile(syn):
            scores[row.label].append(verify_files(verification, gt, syn)[0])
        else:
            print(row.ref, row.syn_trgt)

    eer = EER(torch.tensor(scores[1]), torch.tensor(scores[0]))

    print('\nEER:', eer)