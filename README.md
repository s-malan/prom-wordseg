# TTI Word Segmentation

## Overview
As part of the "What Do Self-Supervised Speech Models Know About Words?" paper, an intrinsic training-free word segmentation algorithm is proposed. The algorithm leverages the behavior of frame-level representations near word segment boundaries. More specifically, the algorithm finds frame-level representations of the input audio from a pre-trained self-supervised speech model. The distance (dissimilarity) between adjacent frames are used (with the addition of smoothing) in conjunction with a prominence-based peak-detection algorithm to predict word boundaries where prominence values exceed some threshold value.

Reference paper: [TTI Paper](https://arxiv.org/abs/2307.00162)

## Pre-Trained Models
- Hubert
  - [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
  - [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/hubert)
  - [bshall](https://github.com/bshall/hubert/tree/main)
- wav2vec 2.0
  - [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)
  - [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)

## Datasets
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
- when using the data, use pre-determined embeddings (they randomly sub-sample ~2k LibriSpeech utterances, thus sample ~2k embedding files)

#### Notes
- import: pytorch audio or librosa (audio manipulation), os (manipulate file structure), json (to save files in json format), fairseq (to load NN checkpoints)
- use bash commands to load data (store in data dictionary)
- to use venv for [HuggingFace](https://huggingface.co/docs/transformers/en/installation):
  - create: python -m venv .venv
  - activate: source .env/bin/activate OR source .venv/bin/activate
  - deactivate: deactivate
  - delete: deactivate AND sudo rm -rf venv