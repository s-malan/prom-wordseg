# TTI Word Segmentation

## Overview
As part of the "What Do Self-Supervised Speech Models Know About Words?" paper, an intrinsic training-free word segmentation algorithm is proposed. The algorithm leverages the behavior of frame-level representations near word segment boundaries. More specifically, the algorithm finds frame-level representations of the input audio from a pre-trained self-supervised speech model. The distance (dissimilarity) between adjacent frames are used (with the addition of smoothing) in conjunction with a prominence-based peak-detection algorithm to predict word boundaries where prominence values exceed some threshold value.

Reference paper: [TTI Paper](https://arxiv.org/abs/2307.00162)

## Pre-Trained Models
**TODO** (HuBERT and wav2vec2, which pre-trained models to use)
- [Hubert](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
- [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)

## Datasets
**TODO**
- [LibriSpeech](https://www.openslr.org/12) Corpus
- Buckeye
- ZeroSpeech 20__

## Example Usage
**TODO** give an example of how to use the repository

## TODO
- eval_seg.py is where functions for all evaluations will be
- extract_seg.py is where main code is to run the word seg algo (not sure if eval code should be called from here or if must run eval_seg.py after algo is run)
- use argparser to run these scripts using command line inputs (paths to data and output repos, also what data and model to use)
- wordseg/model.py is where functions for the word seg algo will be
- wordseg/utils.py is where functions for basic utilities will be (saving (checkpoints and data), loading (checkpoints and data), data processing tasks)
- data/ is where the data will be saved (add in .gitnore)
- add comment blocks in .py files to describe functions (also inputs and outputs) (and one at top of file to explain what is in it)

#### Notes
- import: pytorch audio or librosa (audio manipulation), os (manipulate file structure), json (to save files in json format), fairseq (to load NN checkpoints)
- use bash commands to load data (store in data dictionary)