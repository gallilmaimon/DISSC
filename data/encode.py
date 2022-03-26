import json
import os

import torchaudio
from tqdm import tqdm
from textless.data.speech_encoder import SpeechEncoder

dense_model_name = "hubert-base-ls960"
quantizer_name, vocab_size = "kmeans", 100
base_dir = 'VCTK-Corpus/wav16_silence_trimmed_padded/'
input_files = os.listdir(base_dir)
out_file = 'VCTK-corpus/hubert100/encoded.txt'

for file in tqdm(input_files):
    waveform, sample_rate = torchaudio.load(os.path.join(base_dir, file))

    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_name,
        vocab_size=vocab_size,
        deduplicate=False,
    ).cuda()

    encoded = encoder(waveform.cuda())
    encoded.pop('dense')
    for k in encoded.keys():
        encoded[k] = encoded[k].cpu().numpy().tolist()
    encoded['audio'] = file
    with open(out_file, 'a+') as f:
        f.write(f'{json.dumps(encoded)}\n')