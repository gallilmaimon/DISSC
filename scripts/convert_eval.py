import os
import subprocess
import shutil
from pathlib import Path
from scipy.io.wavfile import write
# import torchaudio
import argparse


def _init_params(args):
    if args.data == 'vctk':
        src_spk = ['p231', 'p239', 'p245', 'p270']
        trgt_spk = ['p231', 'p239', 'p245', 'p270']
        spk_dict = {'p231': '6', 'p239': '13', 'p245': '18', 'p270': '43'}
        wanted_seq = list(range(1, 25))
        gen_path = f'outputs/{args.data}/{args.dissc_type}/'
        data_path = f'data/VCTK/'
        out_path = f'results/{args.data}/{args.dissc_type}/'
        gt_suf = '_mic2.flac'
        pred_suf = '_mic2'
        tf_name = 'val'  # test file name
        gt_pre = ''  # file prefix in wav for test files
    elif args.data == 'syn_vctk':
        src_spk = ['p231', 'p232', 'p233', 'p239', 'p245', 'p270']
        trgt_spk = ['p231', 'p239', 'p245', 'p270']
        spk_dict = {'p231': '6', 'p239': '13', 'p245': '18', 'p270': '43'}
        wanted_seq = list(range(1, 25))
        gen_path = f'outputs/{args.data}/{args.dissc_type}/'
        data_path = f'data/Syn_VCTK/'
        out_path = f'results/{args.data}/{args.dissc_type}/'
        gt_suf = '.wav'
        pred_suf = ''
        tf_name = 'val'  # test file name
        gt_pre = ''  # file prefix in wav for test files
    elif args.data == 'esd':
        src_spk = ['0014Sad', '0015Surprise', '0017Happy', '0019Sad']
        trgt_spk = ['0014Sad', '0015Surprise', '0017Happy', '0019Sad']
        spk_dict = {'0014Sad': '3', '0015Surprise': '4', '0017Happy': '6', '0019Sad': '8'}
        wanted_seq = list(range(1, 35))
        gen_path = f'outputs/{args.data}/{args.dissc_type}/'
        data_path = f'data/ESD/'
        out_path = f'results/{args.data}/{args.dissc_type}/'
        gt_suf = '.wav'
        pred_suf = ''
        tf_name = 'test'  # test file name
        gt_pre = 'paired_test/'  # file prefix in wav for test files
    else:
        print(f"unsupported dataset: {args.data} !")
        exit(1)
    return {'src_spk':src_spk, 'trgt_spk':trgt_spk, 'spk_dict':spk_dict, 'wanted_seq':wanted_seq, 'gen_path':gen_path,
            'data_path':data_path, 'out_path':out_path, 'gt_suf':gt_suf, 'pred_suf':pred_suf, 'tf_name':tf_name,
            'gt_pre':gt_pre}


def _run_bash(command):
    p = subprocess.Popen(command.split())
    p.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='vctk', help='dataset from [vctk, esd, syn_vctk]')
    parser.add_argument('--dissc_type', default='dissc_b', help='conversion type if pitch-only, rhythm-only or both [dissc_p, dissc_l, dissc_b]')
    parser.add_argument('--sort_gt', action='store_true', help='If true we structure the GT as well, this only needs to be done once for each dataset')
    args = parser.parse_args()
    # Constants
    c = _init_params(args)

    # Take only wanted speakers in val
    path = f"{c['data_path']}/hubert100/"
    with open(path + f'{c["tf_name"]}.txt', 'r') as f_in, open(path + f'{c["tf_name"]}_sf.txt', 'w+') as f_out:
        for line in f_in:
            if eval(line)['audio'].split('/')[-1].split('_')[0] in c['src_spk']:
                f_out.write(line)

    # Convert prosody with DISSC
    c1 = f"python3 infer.py --input_path {c['data_path']}/hubert100/{c['tf_name']}_sf.txt --out_path {c['data_path']}/pred_hubert_{args.dissc_type}/ --len_model checkpoints/{args.data}/len/ --f0_model checkpoints/{args.data}/pitch/ --f0_path {c['data_path']}/hubert100/f0_stats.pkl --vc --target_speakers {' '.join(c['trgt_spk'])} -n 1000"
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
    for t_spk in c['trgt_spk']:
        c2 = f"python3 sr/inference.py --input_code_file {c['data_path']}/pred_hubert_{args.dissc_type}/{t_spk}_{c['tf_name']}_sf.txt --data_path {c['data_path']}/wav/{c['gt_pre']} --output_dir {c['gen_path']}/{t_spk} --checkpoint_file sr/checkpoints/{args.data.split('_')[-1]}_hubert --vc --target-speakers {t_spk} -n 1000"
        _run_bash(c2)

    # Reorganise the structure of DISSC
    os.makedirs(Path(c['out_path']).parent.parent, exist_ok=True)
    os.makedirs(Path(c['out_path']).parent, exist_ok=True)
    os.makedirs(Path(c['out_path']), exist_ok=True)
    for t_spk in c['trgt_spk']:
        os.makedirs(f"{c['out_path']}/{t_spk}", exist_ok=True)
        for spk in c['src_spk']:
            for seq in c['wanted_seq']:
                try:
                    shutil.copy(f"{c['gen_path']}/{t_spk}/{spk}_{seq:03}{c['pred_suf']}_{c['spk_dict'][t_spk]}_gen.wav",
                                f"{c['out_path']}/{t_spk}/{spk}_{seq:03}.wav")
                    shutil.copy(f"{c['data_path']}/txt/{spk}/{spk}_{seq:03}.txt", f"{c['out_path']}/{t_spk}/{spk}_{seq:03}.txt")
                except FileNotFoundError:
                    print(f'No sample: {spk}_{seq:03}, this is ok if it only happens for few of the samples because not all speakers have all utterances')

    # Reorganise the structure of the original
    if args.sort_gt:
        gt_path = f'results/{args.data}/orig/'
        os.makedirs(gt_path, exist_ok=True)
        for spk in c['src_spk']:
            for seq in c['wanted_seq']:
                if os.path.exists(f"{c['data_path']}/wav/{c['gt_pre']}/{spk}_{seq:03}{c['gt_suf']}"):
                    gt_sample = torchaudio.load(f"{c['data_path']}/wav/{c['gt_pre']}/{spk}_{seq:03}{c['gt_suf']}")[0][0].numpy()
                    write(f"{gt_path}/{spk}_{seq:03}.wav", 16000, gt_sample)
                    shutil.copy(f"{c['data_path']}/txt/{spk}/{spk}_{seq:03}.txt", f'{gt_path}/{spk}_{seq:03}.txt')
                else:
                    print(f'No sample: {spk}_{seq:03}')

    # # Run MFA to get alignment
    # for t_spk in c['trgt_spk']:
    #     os.makedirs(f"{c['out_path']}/{t_spk}/txtgrid", exist_ok=True)
    #     c3 = f"mfa align -s 4 --clean {c['out_path']}/{t_spk}/ english_us_arpa english_us_arpa {c['out_path']}/{t_spk}/txtgrid/"
    #     _run_bash(c3)
    #     shutil.rmtree(os.path.expanduser(f"~/Documents/MFA/{t_spk}_pretrained_aligner"))
    #
    # if args.sort_gt:
    #     os.makedirs(f"{gt_path}/txtgrid/", exist_ok=True)
    #     c3 = f"mfa align -s 4 --clean {gt_path}/ english_us_arpa english_us_arpa {gt_path}/txtgrid/"
    #     _run_bash(c3)
    #     shutil.rmtree(os.path.expanduser(f"~/Documents/MFA/orig_pretrained_aligner"))

    # Run evaluation
    # c4 = f"python3 eval.py --base_path results/{args.data} --method {args.dissc_type} --target_speakers {' '.join(c['trgt_spk'])}"
    # _run_bash(c4)