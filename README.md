# Speaking Style Conversion With Discrete Self-Supervised Units
Official implementation of ["Speaking Style Conversion With Discrete Self-Supervised Units"](https://arxiv.org/abs/2212.09730).

**** Put Architecture image here ****

__Abstract__: Voice conversion is the task of making a spoken utterance by one speaker sound as if uttered by a different speaker, while keeping other aspects like content the same. Existing methods focus primarily on spectral features like timbre, but ignore the unique speaking style of people which often impacts prosody. In this study we introduce a method for converting not only the timbre, but also the rhythm and pitch changes to those of the target speaker. In addition, we do so in the many-to-many setting with no paired data. We use pretrained, self-supervised, discrete units which make our approach extremely light-weight. We introduce a suite of quantitative and qualitative evaluation metrics for this setup, and show that our approach outperforms existing methods.

## Quick Links
* [Samples](https://pages.cs.huji.ac.il/adiyoss-lab/dissc/)
* [Setup](#setup)
* [Inference (with pretrained models)](#infer)
* [Evaluation (calculate metrics)](#evaluation)
* [Training](#train)

## Setup
We present all setup requirements for running all parts of the code - including evaluation metrics, and all datasets. This adds further limitations which might not be mandatory otherwise, such as Montreal Forced Aligner requiring Conda installation because of Kaldi. You can also install requirements and download datasets by use.

### Environment
Create a conda environment, with python version 3.8 and install all the dependencies:
```sh
conda create -n dissc python=3.8
conda activate dissc

# download code
git clone https://github.com/gallilmaimon/DISSC.git
cd DISSC

# install requirements
conda config --append channels conda-forge  # add needed channels
conda config --append channels pytorch
conda config --append channels nvidia
conda install --file requirements.txt

# install textlesslib, based on https://github.com/facebookresearch/textlesslib#installation
cd ..

```

While certain other versions may be compatible as well, this was only tested with this setup.

### Data
We describe here how to download, preprocess and parse VCTK, Emotional Speech Dataset (ESD) and our synthetic dataset - Syn_VCTK. We assume you are using Linux with default installations for downloading, extracting and deleting files, but this can easily be adapted.

#### For VCTK
1. Download the data from [here](https://datashare.ed.ac.uk/handle/10283/3443) and extract to ```data/VCTK/wav_orig``` folder. This can be done with:
2. Preprocess the audio (downsample audio from 48 kHz to 16 kHz and pad). One could also trim silences to potentially improve results, but we do not do so.
```sh
python3 data/preprocess.py --srcdir data/VCTK/wav_orig --outdir data/VCTK/wav --pad --postfix mic2.flac
```

#### For ESD
1. Download the preprocessed data from [here](https://drive.google.com/file/d/1pX-G5geLLHc0852ZD_YlJwNa8_NKspaL/view?usp=share_link) to ```data/ESD/wav``` folder.
2. If you want to preprocess this dataset from scratch, for instance if you wish to select different emotions for each speaker, download the entire dataset from [here](https://drive.google.com/file/d/1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v/view).

#### For Syn_VCTK
1. Download the preprocessed data from [here](https://drive.google.com/file/d/1xOBGa-t2z8fSTU8aveVgiVsILdNVzvaG/view?usp=share_link) to ```data/Syn_VCTK/wav``` folder.


## Infer
This section discusses how to perform speaking style conversion on a given sample with a trained model (in this case syn_vctk). We show the option of converting a sample of an unseen speaker (in the any-to-many) setup with a sample we recorded ourselves. For converting a subset of data from known speakers (such as the validation set), see the [evaluation](#evaluation) section.

### Any-to-Many
1. Preprocess the recording, resample to 16khz if needed and pad as needed:
```sh
python3 data/preprocess.py --srcdir data/unseen/wav_orig --outdir data/unseen/wav --pad --postfix .wav
```

2. Encode the sample with HuBERT:
```sh
python3 data/encode.py --base_dir data/unseen/wav --out_file data/unseen/hubert100/encoded.txt --device cuda:0
```

3. Download the pretrained models from [here](https://drive.google.com/drive/folders/1oTvW0lxIyrPuEUchfTBSXYpdNMUUXh6n?usp=share_link) to ```checkpoints/syn_vctk``` in the current file structure and all files from [here](https://drive.google.com/drive/folders/1LNP0u35EuBeGmXG5UIjyQnlWS78F2nGm?usp=share_link) to ```sr/checkpoints/vctk_hubert```.

4. Convert the prosody - rhythm (--pred_len option) and pitch contour (--pred_pitch option) using DISSC:
```sh
python3 infer.py --input_path data/unseen/hubert100/encoded.txt --out_path data/unseen/pred_hubert/ --pred_len --pred_pitch --len_model checkpoints/syn_vctk_baseline/len/ --pitch_model checkpoints/syn_vctk_baseline/pitch/ --f0_path data/Syn_VCTK/hubert100/f0_stats.pkl --vc --target_speakers p231 p239 p245 p270
```

5. Resnythesise the audio with speech-resynthesis in the new speaker's voice and style, for here we demonstrate with p231 from Syn_VCTK:
```sh
python3 sr/inference.py --input_code_file data/unseen/hubert100/p231_encoded.txt --data_path data/unseen/wav --out_path dissc_p231 --checkpoint_file sr/checkpoints/vctk_hubert --unseen_speaker --id_to_spkr data/Syn_VCTK/hubert100/id_to_spkr.pkl
```

## Evaluation
This section discusses how to evaluate the pretrained models on each of the datasets, first performing the VC and then calculating all metrics.

### VCTK
0. Download the pretrained model from [here]() to ```results/syn_vctk```.

1. Encode the sample using HuBERT:
```sh
python3 data/encode.py --base_dir data/sample/wav --out_file data/sample/hubert/encoded.txt --device cuda:0
```

2. Convert the prosody - rhythm (--pred_len option), pitch contour (--pred_pitch option) or both, using DISSC:
```sh
python3 infer.py --input_path ... --out_path ... --pred_len --pred_pitch --len_model results/vctk_baseline/len/ --pitch_model results/vctk_baseline/pitch/ --f0_path data/VCTK/hubert100/f0_stats.pkl --vc --target_speakers p245
```

3. Resnythesise the audio with speech-resynthesis in the new speaker's voice and style:
```sh
python3 sr/inference.py ...
```

### ESD
0. Download the pretrained model from [here]() to ```results/syn_vctk```.

1. Encode the sample using HuBERT:
```sh
python3 data/encode.py --base_dir data/sample/wav --out_file data/sample/hubert/encoded.txt --device cuda:0
```

2. Convert the prosody - rhythm (--pred_len option), pitch contour (--pred_pitch option) or both, using DISSC:
```sh
python3 infer.py --input_path ... --out_path ... --pred_len --pred_pitch --len_model results/vctk_baseline/len/ --pitch_model results/vctk_baseline/pitch/ --f0_path data/VCTK/hubert100/f0_stats.pkl --vc --target_speakers p245
```

3. Resnythesise the audio with speech-resynthesis in the new speaker's voice and style:
```sh
python3 sr/inference.py ...
```

### Syn_VCTK
0. Download the pretrained model from [here]() to ```results/syn_vctk```.

1. Encode the sample using HuBERT:
```sh
python3 data/encode.py --base_dir data/sample/wav --out_file data/sample/hubert/encoded.txt --device cuda:0
```

2. Convert the prosody - rhythm (--pred_len option), pitch contour (--pred_pitch option) or both, using DISSC:
```sh
python3 infer.py --input_path ... --out_path ... --pred_len --pred_pitch --len_model results/vctk_baseline/len/ --pitch_model results/vctk_baseline/pitch/ --f0_path data/VCTK/hubert100/f0_stats.pkl --vc --target_speakers p245
```

3. Resnythesise the audio with speech-resynthesis in the new speaker's voice and style:
```sh
python3 sr/inference.py ...
```


## Train
This section discusses how to train the models from scratch.

## Reference
If you found this code useful, we would appreciate you citing the related paper
```bib
@article{maimon2022speaking,
  title={Speaking Style Conversion With Discrete Self-Supervised Units},
  author={Maimon, Gallil and Adi, Yossi},
  journal={arXiv preprint arXiv:2212.09730},
  year={2022}
}
```
