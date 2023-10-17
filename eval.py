import os
import glob
import pickle
from pathlib import Path
import argparse

import torchaudio
import numpy as np

# FFE metrics
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from librosa.util import normalize
import textgrid

# ASR metrics
import string
import whisper
import editdistance as ed

# Earth movers distance
from scipy.stats import wasserstein_distance as emd
from utils import interp


def get_yaapt(audio):
    frame_length = 20.0  # ms
    to_pad = int(frame_length / 1000 * 16000) // 2
    audio = np.pad(audio, (to_pad, to_pad), "constant", constant_values=0)
    audio = normalize(audio) * 0.95
    audio = basic.SignalObj(audio, 16000)
    pitch = pYAAPT.yaapt(audio, frame_length=frame_length, nccf_thresh1=0.25, frame_space=0.005 * 1000, tda_frame_length=25.0)
    return pitch.samp_values


def calc_asr_er(ref, pred):
    int_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                '8': 'eight', '9': 'nine'}
    gt_text = ref.lower().strip().translate(str.maketrans('', '', string.punctuation))
    ref_w = gt_text.split()
    ref_c = list(' '.join(ref_w))
    pred_text = pred.lower().strip().translate(str.maketrans('', '', string.punctuation))
    for k, v in int_dict.items():
        pred_text = pred_text.replace(k, v)
    syn_w = pred_text.split()
    syn_c = list(' '.join(syn_w))
    return ed.eval(syn_c, ref_c), len(ref_c), ed.eval(syn_w, ref_w), len(ref_w)


def aligned_ffe(int1, int2, pitch1, pitch2, sr=16000):
    ffe = []
    for i in range(len(int1)):
        ref = pitch1[int(int1[i].minTime * sr * 0.005 * 2.5 + 2):int(int1[i].maxTime * sr * 0.005 * 2.5 + 2)]
        syn = pitch2[int(int2[i].minTime * sr * 0.005 * 2.5 + 2):int(int2[i].maxTime * sr * 0.005 * 2.5 + 2)]
        syn = interp(syn, ref.shape[0])
        ffe.append((np.abs(((ref + 0.0001)/(syn + 0.0001))-1) > 0.2).mean())
    return np.mean(ffe)


def calc_errors(asr_model, args):
    gt_path = f'{args.base_path}/orig/'
    gen_path = f'{args.base_path}/{args.method}/'
    err_dict = {'wer_s': 0, 'wer_d': 0, 'cer_s': 0, 'cer_d': 0, 'len': [], 'emd': [], 'w_ffe': [], 'w_len': [],
                'p_ffe': [], 'p_len': []}

    for trg in args.target_speakers:
        print(f'--- speaker {trg} -----')
        for f in glob.glob(f'{gen_path}/{trg}/*.wav'):
            if trg in f.split('/')[-1]:  # don't evaluate reconstruction
                continue
            seq = f.split('_')[-1].split('.')[0]

            if not os.path.isfile(f'{gt_path}/{trg}_{seq}.wav'):
                print('No reference recording: ', f'{trg}_{seq}.wav')
                continue
            path = Path(f)
            if path.stem.split('_')[0] == 'p270' and seq == '024':
                print('p270_024 is a problematic sample where content varies notably!')
                continue

            yref, sr = torchaudio.load(f'{gt_path}/{trg}_{seq}.wav')
            ysyn, _sr = torchaudio.load(f)
            yref, ysyn = yref[0], ysyn[0]
            assert sr == _sr, f"{sr} != {_sr}"

            # Length Error
            err_dict['len'].append(abs(len(yref) - len(ysyn)))

            # ASR metrics
            ref_text = open(f'{os.path.splitext(f)[0]}.txt', 'r').readline()
            pred_text = asr_model.transcribe(f)['text']
            res = calc_asr_er(ref_text, pred_text)
            err_dict['cer_s'], err_dict['cer_d'], err_dict['wer_s'], err_dict['wer_d'] = err_dict['cer_s'] + res[0], err_dict['cer_d'] + res[1], err_dict['wer_s'] + res[2], err_dict['wer_d'] + res[3]

            # Earth Movers Distance
            syn_pitch = get_yaapt(ysyn.numpy())
            ref_pitch = get_yaapt(yref.numpy())
            if ref_pitch.shape[0] > syn_pitch.shape[0]:  # Make pitch seqs the same length
                syn_pitch = np.pad(syn_pitch, (0, ref_pitch.shape[0] - syn_pitch.shape[0]), constant_values=0)
            elif yref.shape[0] < ysyn.shape[0]:
                ref_pitch = np.pad(ref_pitch, (0, syn_pitch.shape[0] - ref_pitch.shape[0]), constant_values=0)
            err_dict['emd'].append(emd(syn_pitch, ref_pitch))

            # Forced alignment metrics
            ref_grid = textgrid.TextGrid.fromFile(f'{gt_path}/txtgrid/{trg}_{seq}.TextGrid')
            syn_grid = None  # If the content is corrupted by the conversion MFA doen't manage to align
            if os.path.isfile(path.parent / f'txtgrid/{path.stem}.TextGrid'):
                syn_grid = textgrid.TextGrid.fromFile(path.parent / f'txtgrid/{path.stem}.TextGrid')
            try:
                phone_ref_grid = [f for f in ref_grid[1] if f.mark]
                if syn_grid:
                    phone_syn_grid = [f for f in syn_grid[1] if f.mark]
                else:
                    phone_syn_grid = [textgrid.Interval(ref_grid.maxTime / (len(ref_grid[1]) + 1) * i, ref_grid.maxTime / (len(ref_grid[1]) + 1) * (i + 1), inv.mark) for i, inv in enumerate(ref_grid[1]) if inv.mark]
                err_dict['p_len'].append(np.abs((np.array([i.duration() for i in phone_ref_grid]) - np.array(
                    [i.duration() for i in phone_syn_grid]))).mean())
                err_dict['p_ffe'].append(aligned_ffe(phone_ref_grid, phone_syn_grid, ref_pitch, syn_pitch, sr))
            except ValueError:
                pass
            try:
                word_ref_grid = [f for f in ref_grid[0] if f.mark]
                if syn_grid:
                    word_syn_grid = [f for f in syn_grid[0] if f.mark]
                else:
                    word_syn_grid = [textgrid.Interval(ref_grid.maxTime / (len(ref_grid[0]) + 1) * i, ref_grid.maxTime / (len(ref_grid[0]) + 1) * (i + 1), inv.mark) for i, inv in enumerate(ref_grid[0]) if inv.mark]
                err_dict['w_len'].append(np.abs((np.array([i.duration() for i in word_ref_grid]) - np.array([i.duration() for i in word_syn_grid]))).mean())
                err_dict['w_ffe'].append(aligned_ffe(word_ref_grid, word_syn_grid, ref_pitch, syn_pitch, sr))
            except ValueError:
                pass
    return err_dict


def log_results(err_dict, args, sr=16000):
    with open(f'{args.base_path}/{args.method}_results.pkl', 'wb') as f:
        pickle.dump(err_dict, f)

    print('WER: ', err_dict['wer_s'] / err_dict['wer_d'])
    print('CER: ', err_dict['cer_s'] / err_dict['cer_d'])
    print('EMD: ', np.mean(err_dict['emd']))
    print('Len Error: ', np.mean(err_dict['len']) / sr)

    print('Word Len Error: ', np.mean(err_dict['w_len']))
    print('Char Len Error: ', np.mean(err_dict['p_len']))
    print('Word FFE: ', np.mean(err_dict['w_ffe']))
    print('Character FFE: ', np.mean(err_dict['p_ffe']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='../results/vctk/', help='Base path to all conversion methods')
    parser.add_argument('--method', default='sr', help='Name of conversion type, as in folder name')
    parser.add_argument('--device', default='cuda:0', help='Torch device')
    parser.add_argument('--target_speakers', nargs='+', default=['p231', 'p239', 'p245', 'p270'], help='Target speakers for VC. If none random speakers are used')
    args = parser.parse_args()

    model = whisper.load_model("medium.en")  # used for ASR metrics
    model.eval()
    model = model.to(args.device)

    errs = calc_errors(model, args)
    log_results(errs, args)
