"""
Script to optimize the hyperparameters of the segmentation algorithm. Uses hardcoded models and layers.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

import json
from tqdm import tqdm
import utils
import argparse
from pathlib import Path
from evaluate import eval_segmentation, get_rvalue
from segment import Segmentor

def grid_search_layer(data, norm_embeddings, strict):
    distances = ['euclidean', 'cosine']
    window_sizes = [3, 4, 5, 6, 7] # [2, 3, 4, 5]
    prominences = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    optimal_parameters = {'distance': '', 'window_size': 0, 'prominence': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'rval': 0}
    for prominence in tqdm(prominences, desc='Prominences'):
        for distance in distances:
            for window_size in window_sizes:
                # Segmenting
                segment = Segmentor(distance_type=distance, prominence=prominence, window_size=window_size) # window_size is in frames (1 frame = 20ms for wav2vec2 and HuBERT)
                segment.get_distance(norm_embeddings) # calculate the normalized distance between each embedding in the sequence
                segment.moving_average() # calculate the moving average of the distances

                peaks, _ = segment.peak_detection() # find the peaks in the distances

                # Evaluate
                alignment_end_times = [alignment.end for alignment in data.alignment_data[:]]
                alignment_end_frames = []
                for alignment_times in alignment_end_times:
                    alignment_frames = [data.get_frame_num(end_time) for end_time in alignment_times]
                    alignment_end_frames.append(alignment_frames)
                
                p, r, f = eval_segmentation(peaks, alignment_end_frames, strict=strict, tolerance=1)
                rval = get_rvalue(p, r)
                # if r+(rval)/2 > optimal_parameters['recall'] + (optimal_parameters['rval'])/2:
                if (f+rval)/2 > (optimal_parameters['f1'] + optimal_parameters['rval'])/2: # can also only use r-value to optimize
                    optimal_parameters['distance'] = distance
                    optimal_parameters['window_size'] = window_size
                    optimal_parameters['prominence'] = prominence
                    optimal_parameters['precision'] = p
                    optimal_parameters['recall'] = r
                    optimal_parameters['f1'] = f
                    optimal_parameters['rval'] = rval
                
    return optimal_parameters

def grid_search(embeddings_dir, alignments_dir, align_format, sample_size=2000, strict = True, melspec = False):
    models = ['w2v2_hf', 'w2v2_fs', 'hubert_hf', 'hubert_fs', 'hubert_shall']
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    optimised_paramters = {}

    for model in tqdm(models, desc='Models'):
        optimised_paramters[model] = {}
        for layer in tqdm(layers, desc=f'{model}: Layers'):
            data = utils.Features(root_dir=embeddings_dir, model_name=model, layer=layer, data_dir=alignments_dir, num_files=sample_size, alignment_format=align_format)

            # Embeddings
            sample = data.sample_embeddings() # sample from the feature embeddings
            embeddings = data.load_embeddings(sample) # load the sampled embeddings
            norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

            # Alignments
            alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
            data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files

            hyperparameters = grid_search_layer(data, norm_embeddings, strict)
            optimised_paramters[model][layer] = hyperparameters
        print(optimised_paramters)

    if melspec:
        optimised_paramters["melspec"] = {}
        data = utils.Features(root_dir=embeddings_dir, model_name="melspec", layer=-1, data_dir=alignments_dir, num_files=sample_size, alignment_format=align_format)

        # Embeddings
        sample = data.sample_embeddings() # sample from the feature embeddings
        embeddings = data.load_embeddings(sample) # load the sampled embeddings
        norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

        # Alignments
        alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
        data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files

        hyperparameters = grid_search_layer(data, norm_embeddings, strict)
        optimised_paramters["melspec"] = hyperparameters
        print(optimised_paramters)

    with open("optimized_parameters.json", "w") as outfile:
        json.dump(optimised_paramters, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters.")
    parser.add_argument(
        "embeddings_dir",
        metavar="embeddings-dir",
        help="path to the embeddings directory.",
        type=Path,
    )
    parser.add_argument(
        "alignments_dir",
        metavar="alignments-dir",
        help="path to the alignments directory.",
        type=Path,
    )
    parser.add_argument(
        "sample_size",
        metavar="sample-size",
        help="number of embeddings to sample.",
        type=int,
    )
    parser.add_argument(
        "--align_format",
        help="extension of the alignment files (defaults to .TextGrid).",
        default=".TextGrid",
        type=str,
    )
    parser.add_argument( # optional argument to make the evaluation strict
        '--strict',
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument( # optional argument to additionally optimize melspec features
        '--melspec',
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args() # python3 wordseg/optimize.py /media/hdd/embeddings/librispeech /media/hdd/data/librispeech_alignments 2000 --strict --melspec
                               # python3 wordseg/optimize.py /media/hdd/embeddings/buckeye/test /media/hdd/data/buckeye_alignments/test 2000 --strict --align_format='.txt'

    grid_search(args.embeddings_dir, args.alignments_dir, args.align_format, args.sample_size, args.strict, args.melspec)