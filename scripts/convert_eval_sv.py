import os
import subprocess
import shutil
from pathlib import Path
import argparse
import pickle
import pandas as pd


def _init_params(args):
    if args.data == 'vctk':
        spk = ['p244', 'p236', 'p300', 'p265', 'p288', 'p304', 'p302', 'p334', 'p232', 'p253', 'p286', 'p284', 'p227',
               'p228', 'p317', 'p258', 'p261', 'p329', 'p339', 'p347', 'p272', 'p271', 'p293', 'p308', 'p249', 'p237',
               'p361', 'p252', 'p273', 'p305', 'p274', 'p364', 'p263', 'p298', 'p276', 'p241', 'p260', 'p341', 'p299',
               'p330', 'p256', 'p264', 'p259', 'p374', 'p239', 'p351', 'p311', 's5', 'p282', 'p257', 'p313', 'p307',
               'p310', 'p323', 'p360', 'p363', 'p312', 'p306', 'p343', 'p247', 'p314', 'p292', 'p266', 'p255', 'p230',
               'p287', 'p234', 'p238', 'p250', 'p275', 'p233', 'p262', 'p326', 'p279', 'p345', 'p333', 'p246', 'p254',
               'p243', 'p295', 'p240', 'p248', 'p229', 'p245', 'p376', 'p318', 'p316', 'p268', 'p231', 'p226', 'p285',
               'p294', 'p283', 'p362', 'p251', 'p269', 'p270', 'p297', 'p278', 'p336', 'p281', 'p225', 'p267', 'p303',
               'p340', 'p301', 'p277', 'p335']
        gen_path = f'outputs/{args.data}/sv/{args.dissc_type}/'
        data_path = f'data/VCTK/'
        out_path = f'results/{args.data}/sv/{args.dissc_type}/'
        gt_suf = '_mic2.flac'
        pred_suf = '_mic2'
        tf_name = 'val'  # test file name
        gt_pre = ''  # file prefix in wav for test files
    elif args.data == 'esd':
        spk = ['0019Sad', '0012Happy', '0013Neutral', '0016Angry', '0011Angry', '0018Neutral', '0017Happy', '0020Surprise', '0015Surprise', '0014Sad']
        gen_path = f'outputs/{args.data}/sv/{args.dissc_type}/'
        data_path = f'data/ESD/'
        out_path = f'results/{args.data}/sv/{args.dissc_type}/'
        gt_suf = '.wav'
        pred_suf = ''
        tf_name = 'test'  # test file name
        gt_pre = 'paired_test/'  # file prefix in wav for test files
    else:
        print(f"unsupported dataset: {args.data} !")
        exit(1)
    return {'spk':spk, 'gen_path':gen_path, 'data_path':data_path, 'out_path':out_path, 'gt_suf':gt_suf,
            'pred_suf':pred_suf, 'tf_name':tf_name, 'gt_pre':gt_pre}


def _run_bash(command):
    p = subprocess.Popen(command.split())
    p.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='vctk', help='dataset from [vctk, esd, syn_vctk]')
    parser.add_argument('--dissc_type', default='dissc_b', help='conversion type if pitch-only, rhythm-only or both [dissc_p, dissc_l, dissc_b]')
    args = parser.parse_args()
    # Constants
    c = _init_params(args)
    df = pd.read_csv(c['data_path'] + 'speaker_verification.csv')
    with open(c['data_path'] + '/hubert100/id_to_spkr.pkl', 'rb') as f:
        id2spkr = pickle.load(f)
    spkr2id = {n: i for i, n in enumerate(id2spkr)}

    # Convert prosody with DISSC
    c1 = f"python3 infer.py --input_path {c['data_path']}/hubert100/{c['tf_name']}.txt --out_path {c['data_path']}/pred_hubert_sv_{args.dissc_type}/ --len_model checkpoints/{args.data}/len/ --f0_model checkpoints/{args.data}/pitch/ --f0_path {c['data_path']}/hubert100/f0_stats.pkl --vc --sample_df {c['data_path']}/speaker_verification.csv -n 100000"
    if args.dissc_type == 'dissc_l' or args.dissc_type == 'dissc_b':
        c1 += " --pred_len"
    if args.dissc_type == 'dissc_p' or args.dissc_type == 'dissc_b':
        c1 += " --pred_pitch"
    if args.data == 'vctk' or args.data == 'esd':
        c1 += ' --f0_model_type base'
    else:
        c1 += ' --f0_model_type new'
    _run_bash(c1)

    # Resynthesise with SR
    os.makedirs(Path(c['gen_path']).parent.parent, exist_ok=True)
    os.makedirs(Path(c['gen_path']).parent, exist_ok=True)
    os.makedirs(Path(c['gen_path']), exist_ok=True)
    for t_spk in c['spk']:
        c2 = f"python3 sr/inference.py --input_code_file {c['data_path']}/pred_hubert_sv_{args.dissc_type}/{t_spk}_{c['tf_name']}.txt --data_path {c['data_path']}/wav/{c['gt_pre']} --output_dir {c['gen_path']}/{t_spk} --checkpoint_file sr/checkpoints/{args.data.split('_')[-1]}_hubert --vc --target-speakers {t_spk} --sample_df {c['data_path']}/speaker_verification.csv -n 1000"
        _run_bash(c2)

    # Reorganise the structure of DISSC
    os.makedirs(Path(c['out_path']).parent.parent, exist_ok=True)
    os.makedirs(Path(c['out_path']).parent, exist_ok=True)
    os.makedirs(Path(c['out_path']), exist_ok=True)
    shutil.copy(f"{c['data_path']}/speaker_verification.csv", f"{Path(c['out_path']).parent.parent}")
    for _, row in df.iterrows():
        os.makedirs(c['out_path'] + row.syn_trgt, exist_ok=True)
        try:
            shutil.copy(f"{c['gen_path']}/{row.syn_trgt}/{row.syn_sample}{c['pred_suf']}_{spkr2id[row.syn_trgt]}_gen.wav",
                        f"{c['out_path']}/{row.syn_trgt}/{row.syn_sample}.wav")
        except FileNotFoundError:
            print(row.syn_sample)

    # Run evaluation
    c3 = f"python3 eval_sv.py --base_path results/{args.data} --method {args.dissc_type} --gt_path {c['data_path']}/wav/{c['gt_pre']} --file_suffix {c['gt_suf']}"
    _run_bash(c3)