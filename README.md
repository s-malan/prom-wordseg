# Prominence-Based Word Segmentation

As part of the "What Do Self-Supervised Speech Models Know About Words?" paper [https://arxiv.org/abs/2307.00162](https://arxiv.org/abs/2307.00162), an intrinsic training-free word segmentation algorithm is proposed. The algorithm leverages the behavior of frame-level representations near word segment boundaries. More specifically, the algorithm finds frame-level representations of the input audio from a pre-trained self-supervised speech model. The distance (dissimilarity) between adjacent frames are used (with the addition of smoothing) in conjunction with a prominence-based peak-detection algorithm to predict word boundaries where prominence values exceed some threshold value.

## Preliminaries

**Pre-Process Speech Data**

Use VAD to extract utterances from long speech files (specifically for ZeroSpeech and BuckEye) by cloning and following the recipes in the repository at [https://github.com/s-malan/data-process](https://github.com/s-malan/data-process).

**Encode Utterances**

Use pre-trained speech models or signal processing methods to encode speech utterances. Example code can be found here [https://github.com/bshall/hubert/blob/main/encode.py](https://github.com/bshall/hubert/blob/main/encode.py) to use HuBERT base for self-supervised audio encoding.
Save the encodings as .npy files with the file path as: 

    model_name/layer_#/relative/path/to/input/audio.npy

where # is replaced with an integer of the self-supervised models' layer used for encoding, and as:

    model_name/relative/path/to/input/audio.npy

when signal processing methods like MFCCs are used.

<!-- ## Example Usage

Get embeddings for each layer of a specified model:

    python3 wordseg/encode.py model_name path/to/audio path/to/encodings/save --extension=.flac

The model_name can be one of: w2v2_fs, w2v2_hf, hubert_fs, hubert_hf, hubert_shall, melspec. The optional extension argument is the extension of the audio files to be processed. -->

## Scripts

<!-- **Optimize hyperparameters for the word segmentation algorithm**

Feed the algorithm some sample utterances to determine the hyperparameters using a grid-search over the three parameters: (1) distance function, (2) average window length, and (3) prominence threshold value.

    python3 wordseg/optimize.py path/to/encodings path/to/audio/alignments num_samples --strict=True --sig_proc=True

The num_samples argument can be set to -1 to sample all the possible audio files in the directory provided. The optional strict argument determines if the word boundary hit count is strict or lenient as decribed by D. Harwath in [this](https://ieeexplore.ieee.org/abstract/document/10022827) paper. The optional sig_proc argument additionally optimizes MFCC or LogMelSpectogram features. -->

**Run the word segmentation algorithm**

    python3 extract_seg.py model_name layer_num path/to/embeddings path/to/audio/alignments num_samples num_samples --align_format=.TextGrid --save_out=path/to/output --load_hyperparams=path/to/parameters.json --strict=True

The **layer_num** argument is a valid layer in the specified model, for signal processing models this value must be -1. The **num_samples** argument can be set to -1 to sample all the possible audio files in the directory provided. The optional **align_format** argument specifies the extension type of the alignment files. The optional **load_hyperparams** argument loads the saved hyperparameters from a JSON file with fields as described below, otherwise the user is prompted to provide hyperparameter values. The optional **strict** argument determines if the word boundary hit count is strict or lenient as decribed by D. Harwath in [this](https://ieeexplore.ieee.org/abstract/document/10022827) paper.

Parameter JSON file structure (# is the layer number, omitted when a signal processing model is used):

    {   "model_name": {
            "#": {
                "distance": "euclidean", # euclidean or cosine
                "window_size": 6, # int
                "prominence": 0.4, # float
            }, etc.
        }, etc. 
    }

**Evaluation**

TODO link my evaluation repo and mention the zrc repo

<!-- ## Pre-Trained Speech Models

- Hubert
  - [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
  - [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/hubert)
  - [bshall](https://github.com/bshall/hubert/tree/main)
- wav2vec 2.0
  - [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)
  - [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) -->

## Datasets

- [ZeroSpeech](https://download.zerospeech.com/) Challenge Corpus (Track 2).
- [LibriSpeech](https://www.openslr.org/12) Corpus (Dev-Clean split) with alignments found [here](https://zenodo.org/records/2619474).
- [Buckeye](https://buckeyecorpus.osu.edu/) Corpus with splits found [here](https://github.com/kamperh/vqwordseg?tab=readme-ov-file#about-the-buckeye-data-splits) and alignments found [here](https://github.com/kamperh/vqwordseg/releases/tag/v1.0).

<!-- ## Results

Results for each model's best performing layer evaluated on each dataset.

### LibriSpeech (Dev-Clean)

#### wav2vec2.0

##### fairseq (layer 7)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 40.85%
    Recall: 47.73%
    F1-score: 44.02%
    R-value: 48.10%
    ---------------------------------------------------------------------------

##### HuggingFace (layer 7)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 42.93%
    Recall: 44.26%
    F1-score: 43.58%
    R-value: 51.28%
    ---------------------------------------------------------------------------

#### HuBERT

##### fairseq (layer 9)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 45.29%
    Recall: 44.66%
    F1-score: 44.97%
    R-value: 53.25%
    ---------------------------------------------------------------------------

##### HuggingFace (layer 9)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 45.29%
    Recall: 44.66%
    F1-score: 44.97%
    R-value: 53.25%
    ---------------------------------------------------------------------------

##### bshall (layer 9, 10)

    ---------------------------------------------------------------------------
    layer 9:
      Word boundaries:
      Precision: 44.59%
      Recall: 46.34%
      F1-score: 45.45%
      R-value: 52.74%

    layer 10:
      Word boundaries:
      Precision: 45.29%
      Recall: 45.02%
      F1-score: 45.15%
      R-value: 53.28%
    ---------------------------------------------------------------------------

#### melspec

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 27.48%
    Recall: 24.85%
    F1-score: 26.10%
    R-value: 38.94%
    ---------------------------------------------------------------------------

### BuckEye (Val/Dev)

#### wav2vec2.0

##### fairseq (layer 7)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 35.66%
    Recall: 37.13%
    F1-score: 36.38%
    R-value: 44.82%
    ---------------------------------------------------------------------------

##### HuggingFace (layer 7)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 42.22%
    Recall: 39.49%
    F1-score: 40.81%
    R-value: 50.46%
    ---------------------------------------------------------------------------

#### HuBERT

##### fairseq (layer 9)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 42.20%
    Recall: 33.72%
    F1-score: 37.49%
    R-value: 49.04%
    ---------------------------------------------------------------------------

##### HuggingFace (layer 9)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 42.20%
    Recall: 33.72%
    F1-score: 37.49%
    R-value: 49.04%
    ---------------------------------------------------------------------------

##### bshall (layer 9, 10)

    ---------------------------------------------------------------------------
    layer 9:
      Word boundaries:
      Precision: 46.66%
      Recall: 34.76%
      F1-score: 39.84%
      R-value: 50.92%

    layer 10:
      Word boundaries:
      Precision: 48.07%
      Recall: 33.44%
      F1-score: 39.44%
      R-value: 50.63%
    ---------------------------------------------------------------------------

#### melspec

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 26.71%
    Recall: 23.96%
    F1-score: 25.26%
    R-value: 38.39%
    ---------------------------------------------------------------------------

### BuckEye (Test)

#### wav2vec2.0

##### fairseq (layer 7)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 33.91%
    Recall: 36.98%
    F1-score: 35.38%
    R-value: 42.69%
    ---------------------------------------------------------------------------

##### HuggingFace (layer 7)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 42.54%
    Recall: 41.17%
    F1-score: 41.85%
    R-value: 50.88%
    ---------------------------------------------------------------------------

#### HuBERT

##### fairseq (layer 9)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 42.00%
    Recall: 34.92%
    F1-score: 38.13%
    R-value: 49.33%
    ---------------------------------------------------------------------------

##### HuggingFace (layer 9)

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 42.00%
    Recall: 34.92%
    F1-score: 38.13%
    R-value: 49.33%
    ---------------------------------------------------------------------------

##### bshall (layer 9, 10)

    ---------------------------------------------------------------------------
    layer 9:
      Word boundaries:
      Precision: 46.40%
      Recall: 36.04%
      F1-score: 40.57%
      R-value: 51.41%

    layer 10:
      Word boundaries:
      Precision: 47.31%
      Recall: 34.29%
      F1-score: 39.76%
      R-value: 50.88%
    ---------------------------------------------------------------------------

#### melspec

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 25.83%
    Recall: 23.47%
    F1-score: 24.59%
    R-value: 37.64%
    --------------------------------------------------------------------------- -->

<!-- #### Notes
- to use venv for [HuggingFace](https://huggingface.co/docs/transformers/en/installation):
  - create: python -m venv .venv
  - activate: source .env/bin/activate OR source .venv/bin/activate
  - deactivate: deactivate
  - delete: deactivate AND sudo rm -rf venv 
- optimize used for other datasets
  - r_rval_copy is buckeye test
  - r_rval is ZS2017 train
  - r_rval_ls is librispeech dev clean 
    - other: {'hubert_shall': {9: {'distance': 'cosine', 'window_size': 5, 'prominence': 0.2, 'precision': 0.4067757636147626, 'recall': 0.5210944915392923, 'f1': 0.4568927789934354, 'rval': 0.45368183473994717}, 10: {'distance': 'cosine', 'window_size': 5, 'prominence': 0.2, 'precision': 0.4039373990808595, 'recall': 0.5337611607142857, 'f1': 0.45986227180814204, 'rval': 0.4383887500492858}}}-->