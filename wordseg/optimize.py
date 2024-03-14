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

def grid_search_layer(data, norm_embeddings):
    distances = ['euclidean', 'cosine']
    window_sizes = [3, 4, 5, 6, 7]
    prominences = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    optimal_parameters = {'distance': '', 'window_size': 0, 'prominence': 0, 'f1': 0, 'rval': 0}
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
                    
                p, r, f = eval_segmentation(peaks, alignment_end_frames)
                rval = get_rvalue(p, r)
                if (f+rval)/2 > (optimal_parameters['f1'] + optimal_parameters['rval'])/2:
                    optimal_parameters['distance'] = distance
                    optimal_parameters['window_size'] = window_size
                    optimal_parameters['prominence'] = prominence
                    optimal_parameters['f1'] = f
                    optimal_parameters['rval'] = rval
                
    return optimal_parameters

def grid_search(embeddings_dir, alignments_dir, sample_size=2000):   
    models = ['w2v2_hf', 'w2v2_fs', 'hubert_hf', 'hubert_fs', 'hubert_shall']
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    optimised_paramters = {}
    for model in tqdm(models, desc='Models'):
        optimised_paramters[model] = {}
        for layer in tqdm(layers, desc=f'{model}: Layers'): # TODO maybe sample once and just change model and layer in the sample paths and then load the new embeddings
            data = utils.Features(root_dir=embeddings_dir, model_name=model, layer=layer, data_dir=alignments_dir, num_files=sample_size)

            # Embeddings
            sample = data.sample_embeddings() # sample from the feature embeddings
            embeddings = data.load_embeddings(sample) # load the sampled embeddings
            norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

            # Alignments
            alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
            data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files

            hyperparameters = grid_search_layer(data, norm_embeddings)
            optimised_paramters[model][layer] = hyperparameters
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

    args = parser.parse_args() #python3 wordseg/optimize.py /media/hdd/embeddings /media/hdd/data/librispeech_alignments 2000

    grid_search(args.embeddings_dir, args.alignments_dir, args.sample_size)