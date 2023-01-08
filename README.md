# Speaking Style Conversion With Discrete Self-Supervised Units
Official implementation of ["Speaking Style Conversion With Discrete Self-Supervised Units"](https://arxiv.org/abs/2212.09730).

**** Put Architecture image here ****

__Abstract__: Voice conversion is the task of making a spoken utterance by one speaker sound as if uttered by a different speaker, while keeping other aspects like content the same. Existing methods focus primarily on spectral features like timbre, but ignore the unique speaking style of people which often impacts prosody. In this study we introduce a method for converting not only the timbre, but also the rhythm and pitch changes to those of the target speaker. In addition, we do so in the many-to-many setting with no paired data. We use pretrained, self-supervised, discrete units which make our approach extremely light-weight. We introduce a suite of quantitative and qualitative evaluation metrics for this setup, and show that our approach outperforms existing methods.

## Quick Links
* [Samples](https://pages.cs.huji.ac.il/adiyoss-lab/dissc/)
* [Setup](#setup)
* [Inference (with pretrained models)](#infer)
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
conda install --file requirements.txt
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
This section discusses how to perform speaking style conversion on a given sample with a trained model. We demonstrate this on a single sample from Syn_VCTK, but this can be adapted to any sample.
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

## Evaluation
This section discusses how to evaluate the pretrained models on each of the datasets.

## Train
This section discusses how to train the models from scratch/

