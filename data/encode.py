import json
import os
from pathlib import Path
import argparse
import torchaudio
from tqdm import tqdm
from textless.data.speech_encoder import SpeechEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='hubert-base-ls960', help='Name for pretrained dense model name')
    parser.add_argument('--quantizer_name', default='kmeans', help='Name for quantising the hidden units')
    parser.add_argument('--vocab_size', default=100, type=int, help='number of unique HuBERT clusters to used')
    parser.add_argument('--base_dir', default='/cs/dataset/Download/adiyoss/gallil_dset/esd/ESD_filtered_trimmed/train', help='Input audio file path')
    parser.add_argument('--out_file', default='ESD/hubert100/train.txt', help='Output path')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')

    args = parser.parse_args()

    encoder = SpeechEncoder.by_name(dense_model_name=args.model_name, quantizer_model_name=args.quantizer_name,
                                    vocab_size=args.vocab_size, deduplicate=False).to(args.device)

    os.makedirs(Path(args.out_file).parent.parent.absolute(), exist_ok=True)
    os.makedirs(Path(args.out_file).parent.absolute(), exist_ok=True)

    input_files = os.listdir(args.base_dir)
    for file in tqdm(input_files):
        waveform, sample_rate = torchaudio.load(os.path.join(args.base_dir, file))

        try:
            encoded = encoder(waveform.to(args.device))
        except IndexError:
            print(f"\nProblem calculating YAAPT for sample {file}")
            continue
        encoded.pop('dense')
        for k in encoded.keys():
            encoded[k] = encoded[k].cpu().numpy().tolist()
        encoded['audio'] = file
        with open(args.out_file, 'a+') as f:
            f.write(f'{json.dumps(encoded)}\n')