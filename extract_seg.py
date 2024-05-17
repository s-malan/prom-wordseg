"""
Main script to extract embeddings, segment the audio, and evaluate the resulting segmentation.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

from wordseg import utils, segment, evaluate
import numpy as np
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

def get_embeddings(data):
    """
    Samples, loads and normalizes embeddings of the audio data.

    Parameters
    ----------
    data : Features Class
        Object containing all information to find alignments for the selected embeddings

    Returns
    -------
    sample : numpy.ndarray
        List of file paths to the sampled embeddings
    embeddings : list
        List of embeddings loaded from the file paths
    norm_embeddings : list
        The normalized feature embeddings
    """

    sample = data.sample_embeddings() # sample from the feature embeddings
    embeddings = data.load_embeddings(sample) # load the sampled embeddings
    norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

    index_del = []
    for i, embedding in enumerate(norm_embeddings): # delete embeddings with only one frame
        if embedding.shape[0] == 1:
            index_del.append(i)

    for i in sorted(index_del, reverse=True):
        del sample[i]
        del embeddings[i]
        del norm_embeddings[i]
    
    if len(sample) == 0:
        print('No embeddings to segment, sampled a file with only one frame.')
        exit()
    
    return sample, embeddings, norm_embeddings

def get_word_segments(norm_embeddings, distance_type="euclidean", prominence=0.6, window_size=5):
    """
    Implements the word segmentation algorighm by finding peaks in the distances between feature embeddings

    Parameters
    ----------
    norm_embeddings : numpy.ndarray
        The normalized feature embeddings
    distance_type : String
        The type of the distance metric used ('euclidean' or 'cosine')
    prominence : float
        Prominence value threshold for peak detection
    window_size : int
        The window size for the moving average (in number of frames (1 frame = 20ms for w2v2 and HuBERT))

    Returns
    -------
    peaks : list (int)
        The frame indices of the detected peaks
    prominences : list (float)
        The prominence values of the detected peaks
    segmentor: Segmentor Class
        The object containing all hyperparameters and methods to segment the embeddings into words
    """

    # get the distances between adjacent frames in the embeddings
    segmentor = segment.Segmentor(distance_type=distance_type, prominence=prominence, window_size=window_size)
    segmentor.get_distance(norm_embeddings)

    # get the moving average of the distances
    segmentor.moving_average()

    # get the peaks in the distances
    peaks, prominences = segmentor.peak_detection()
    return peaks, prominences, segmentor

def set_alignments(data, sample):
    """
    Sets the text, start and end attributes of the alignment files corresponding to the sampled embeddings

    Parameters
    ----------
    data : Features Class
        Object containing all information to find alignments for the selected embeddings
    sample : list
        List of file paths to the sampled embeddings
    """

    alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
    data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files

if __name__ == "__main__":
    def plot_seg(segment, data, peaks):
        """
        Plots the distances, moving average, hypothesized peaks and ground truth alignments for the first file in the sample 

        Parameters
        ----------
        segment : Segmentor Class
            Object containing all hyperparameters and methods to segment the embeddings into words
        data : Features Class
            Object containing all information to find alignments for the selected embeddings
        peaks : list (int)
            The frame indices of the detected peaks
        """

        peaks = peaks[0]
        
        _, ax = plt.subplots()
        ax.plot(segment.distances[0], label='Distances', color='blue', alpha=0.5)
        # ax.plot(np.range(segment.distances[0]), segment.distances[0] label='Distances', color='blue')
        ax.plot(segment.smoothed_distances[0], label='Smooth Distances', color='red', alpha=0.5)
        ax.scatter(peaks, segment.smoothed_distances[0][peaks], marker='x', label='Peaks', color='green')

        alignment_end_times = data.alignment_data[0].end
        alignment_end_frames = [data.get_frame_num(end_time) for end_time in alignment_end_times]
        print('Segment end times, frames, and text')
        print(alignment_end_times)
        print(alignment_end_frames)
        print(data.alignment_data[0].text)

        for frame in alignment_end_frames:
            ax.axvline(x=frame, label='Ground Truth', color='black', linewidth=0.5)
            ax.axvline(x=frame-1, color='black', linewidth=0.2, alpha=0.5)
            ax.axvline(x=frame+1, color='black', linewidth=0.2, alpha=0.5)

        custom_ticks = alignment_end_frames
        custom_tick_labels = data.alignment_data[0].text
        ax.set_xlim(xmin=0)
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_tick_labels, rotation=90, fontsize=6)

        plt.savefig('distances.png', dpi=300)
        
    parser = argparse.ArgumentParser(description="Segment speech audio.")
    parser.add_argument(
        "model",
        help="available models (wav2vec2.0: HuggingFace, fairseq, hubert: HuggingFace, fairseq, hubert:main)",
        choices=["w2v2_hf", "w2v2_fs", "hubert_hf", "hubert_fs", "hubert_shall", "melspec"],
        default="w2v2_hf",
    )
    parser.add_argument(
        "layer", # -1 for no layer
        type=int,
    )
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
        help="number of embeddings to sample (-1 to sample all available data).",
        type=int,
    )
    parser.add_argument(
        "--align_format",
        help="extension of the alignment files (defaults to .TextGrid).",
        default=".TextGrid",
        type=str,
    )
    parser.add_argument(
        "--save_out",
        help="option to save the output segment boundaries in ms to the specified directory.",
        default=None,
        type=Path,
    )
    parser.add_argument( # optional argument to load the optimized hyperparameters from a file
        '--load_hyperparams',
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument( # optional argument to make the evaluation strict
        '--strict',
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args() #python3 extract_seg.py w2v2_hf 12 /media/hdd/embeddings/librispeech /media/hdd/data/librispeech_alignments -1 --load_hyperparams --strict
    # python3 extract_seg.py w2v2_hf 12 /media/hdd/embeddings/buckeye/dev /media/hdd/data/buckeye_alignments/dev -1 --align_format=.txt --load_hyperparams --strict
    # python3 extract_seg.py hubert_shall 10 /media/hdd/embeddings/buckeye/test /media/hdd/data/buckeye_alignments/test -1 --align_format=.txt --save_out=/media/hdd/segments/tti_wordseg/buckeye/test --load_hyperparams --strict
    # python3 extract_seg.py hubert_shall 10 /media/hdd/embeddings/zrc/zrc2017_train_segments/english /media/hdd/data/zrc_alignments/zrc2017_train_alignments/english -1 --align_format=.txt --save_out=/media/hdd/segments/tti_wordseg/zrc2017_train_segments/english --load_hyperparams --strict

    if not args.load_hyperparams: # ask for hyperparameters
        print("Enter the hyperparameters for the segmentation algorithm: ")
        dist = str(input("Distance metric (euclidean, cosine): "))
        window = int(input("Moving average window size (int): "))
        prom = float(input("Peak detection prominence value (float): "))
    else: #load from file
        with open('optimized_parameters.json') as json_file:
            params = json.load(json_file)
            dataset_name = args.alignments_dir.stem
            if args.layer != -1:
                params = params[args.model][str(args.layer)]
            else:
                params = params[args.model]
            dist = params['distance']
            window = params['window_size']
            prom = params['prominence']
    
    data = utils.Features(root_dir=args.embeddings_dir, model_name=args.model, layer=args.layer, data_dir=args.alignments_dir, alignment_format=args.align_format, num_files=args.sample_size)

    # Embeddings
    sample, embeddings, norm_embeddings = get_embeddings(data)
    
    # Segmenting
    peaks, prominences, segmentor = get_word_segments(norm_embeddings, distance_type=dist, prominence=prom, window_size=window)

    # Optionally save the output segment boundaries
    if args.save_out is not None:
        root_save = args.save_out / args.model / str(args.layer)
        for i, (peak, file) in enumerate(zip(peaks, sample)):
            if len(peak) == 0:
                peak = np.array([embeddings[i].shape[0]]) # add a peak at the end of the file
            peak = data.get_sample_second(peak) # get peak to seconds
            save_dir = (root_save / os.path.split(file)[-1]).with_suffix(".list")
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            with open(save_dir, "w") as f: # save the landmarks to a file
                for l in peak:
                    f.write(f"{l}\n")

    # Alignments
    set_alignments(data, sample)

    # Evaluate
    alignment_end_frames = [alignment.end for alignment in data.alignment_data[:]]

    p, r, f = evaluate.eval_segmentation(peaks, alignment_end_frames, strict=args.strict)
    rval = evaluate.get_rvalue(p, r)
    print('Evaluation: \n Precision: ', p, '\n Recall: ', r, '\n F1-Score: ', f, '\n R-Value: ', rval)
    
    # Plot the distances, moving average and peaks AND compare to the alignments of a single file
    plot_seg(segmentor, data, peaks)