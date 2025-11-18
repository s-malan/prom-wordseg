# Prominence-Based Word Segmentation

As part of the "What Do Self-Supervised Speech Models Know About Words?" paper [https://arxiv.org/abs/2307.00162](https://arxiv.org/abs/2307.00162), an intrinsic training-free word segmentation algorithm is proposed. The algorithm leverages the behavior of frame-level representations near word segment boundaries. More specifically, it finds frame-level representations of the input audio from a pre-trained self-supervised speech model. The distance (dissimilarity) between adjacent frames are used (with the addition of smoothing) in conjunction with a prominence-based peak-detection algorithm to predict word boundaries where prominence values exceed some threshold value.

## Preliminaries

**Datasets**

- [ZeroSpeech](https://download.zerospeech.com/) Challenge Corpus (Track 2).
- [LibriSpeech](https://www.openslr.org/12) Corpus (Dev-Clean split) with alignments found [here](https://zenodo.org/records/2619474).
- [Buckeye](https://buckeyecorpus.osu.edu/) Corpus with splits found [here](https://github.com/kamperh/vqwordseg?tab=readme-ov-file#about-the-buckeye-data-splits) and alignments found [here](https://github.com/kamperh/vqwordseg/releases/tag/v1.0).

**Pre-Process Speech Data**

Use VAD to extract utterances from long speech files (specifically for ZeroSpeech and BuckEye) by cloning and following the recipes in the repository at [https://github.com/s-malan/data-process](https://github.com/s-malan/data-process).

**Encode Utterances**

Use pre-trained speech models or signal processing methods to encode speech utterances. Example code can be found here [https://github.com/bshall/hubert/blob/main/encode.py](https://github.com/bshall/hubert/blob/main/encode.py) using HuBERT-base for self-supervised audio encoding.
Save the feature encodings as .npy files with the file path as: 

    model_name/layer_#/relative/path/to/input/audio.npy

where # is replaced with an integer of the self-supervised model layer used for encoding, and as:

    model_name/relative/path/to/input/audio.npy

when signal processing methods like MFCCs are used.

## Scripts

**Run the word segmentation algorithm**

    python3 extract_seg.py model_name layer_num path/to/encodings num_samples --save_out=path/to/output --load_hyperparams=path/to/parameters.json --batch_size=25000

The `layer_num` argument is a valid layer in the specified model, for signal processing models this value must be `-1`. The `num_samples` argument can be set to `-1` to sample all the possible audio files in the directory provided. The optional `load_hyperparams` argument loads the saved hyperparameters from a JSON file with fields as described below, otherwise the user is prompted to provide hyperparameter values.  The optional `batch_size` argument specifies how many utterances should be processed at once, the default is `25000`.

Hyperparameter JSON file structure (# is the layer number and (the block) is omitted when a signal processing model with no layers is used):

    {   "model_name": {
            "#": {
                "distance": "euclidean", # euclidean or cosine
                "window_size": 6, # int
                "prominence": 0.4, # float
            }, etc.
        }, etc. 
    }

## Evaluation

To evaluate the resultant hypothesized word boundaries, refer to: [https://github.com/s-malan/prom-seg-clus/blob/main/README.md#evaluation](https://github.com/s-malan/prom-seg-clus/blob/main/README.md#evaluation).