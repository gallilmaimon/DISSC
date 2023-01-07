# Speaking Style Conversion With Discrete Self-Supervised Units
Official implementation of ["Speaking Style Conversion With Discrete Self-Supervised Units"](https://arxiv.org/abs/2212.09730).

**** Put Architecture image here ****

__Abstract__: Voice conversion is the task of making a spoken utterance by one speaker sound as if uttered by a different speaker, while keeping other aspects like content the same. Existing methods focus primarily on spectral features like timbre, but ignore the unique speaking style of people which often impacts prosody. In this study we introduce a method for converting not only the timbre, but also the rhythm and pitch changes to those of the target speaker. In addition, we do so in the many-to-many setting with no paired data. We use pretrained, self-supervised, discrete units which make our approach extremely light-weight. We introduce a suite of quantitative and qualitative evaluation metrics for this setup, and show that our approach outperforms existing methods.

## Quick Links
* [Samples](https://pages.cs.huji.ac.il/adiyoss-lab/dissc/)
* [Setup](#setup)
* Inference (with pretrained models)
* Training

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
1. Download the data from [here](https://datashare.ed.ac.uk/handle/10283/3443) to ```data/VCTK``` folder. This can be done with:
```sh
# VCTK
wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip # This could take a while
mkdir data/VCTK
unzip DS_10283_3443.zip -d data/VCTK
mv data/VCTK/DS_10283_3443 data/VCTK/wav16
rm -rf DS_10283_3443.zip  # cleanup
```
2. Preprocess the audio (downsample audio from 48 kHz to 16 kHz and pad). One could also trim silences to potentially improve results, but we do not do so.
```sh
python3 preprocess.py ...
```


#### For ESD
```sh
# ESD

```


#### For Syn_VCTK
```sh
# Syn_VCTK
# Download large file from drive, can also be done manually
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G3NbgTqZDVc_AXfc5o6BkiNuZQWY6AZy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1G3NbgTqZDVc_AXfc5o6BkiNuZQWY6AZy" -O syn_vctk.tar.gz && rm -rf /tmp/cookies.txt

mkdir data/Syn_VCTK
tar -xzvf syn_vctk.tar.gz -C data/Syn_VCTK
rm -rf syn_vctk.tar.gz  # cleanup
```
