"""
Main script to extract unsupervised prominence based word boundaries.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

from wordseg import utils, segment
from tqdm import tqdm
import numpy as np
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchaudio
from glob import glob
import itertools

def get_features(data, batch_size, batch_num):
    """
    Samples, loads and normalizes features of the audio data.

    Parameters
    ----------
    data : Features Class
        Object containing all information to find alignments for the selected features

    Returns
    -------
    sample : numpy.ndarray
        List of file paths to the sampled features
    features : list
        List of features loaded from the file paths
    norm_features : list
        The normalized feature features
    """

    sample = data.sample_features() # sample from the features
    if batch_size*(batch_num+1) <= len(sample):
        sample = sample[batch_size*batch_num:batch_size*(batch_num+1)]
        batch = True
    else:
        sample = sample[batch_size*batch_num:]
        batch = False
    batch_num = batch_num + 1

    features = data.load_features(sample) # load the sampled features
    norm_features = data.normalize_features(features) # normalize the sampled features

    index_del = []
    for i, norm_feature in enumerate(norm_features): # delete features with only one frame
        if norm_feature.shape[0] == 1:
            index_del.append(i)
    
    if len(sample) == 0:
        print('No features to segment, sampled a file with only one frame.')
        exit()
    
    return sample, features, norm_features, index_del, batch, batch_num

def get_word_segments(norm_features, distance_type="euclidean", prominence=0.6, window_size=5):
    """
    Implements the word segmentation algorighm by finding peaks in the distances between features

    Parameters
    ----------
    norm_features : numpy.ndarray
        The normalized feature features
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
        The object containing all hyperparameters and methods to segment the features into words
    """

    # get the distances between adjacent frames in the features
    segmentor = segment.Segmentor(distance_type=distance_type, prominence=prominence, window_size=window_size)
    segmentor.get_distance(norm_features)

    # get the moving average of the distances
    segmentor.moving_average()

    # get the peaks in the distances
    peaks, prominences = segmentor.peak_detection()
    return peaks, prominences, segmentor

def set_alignments(data, sample):
    """
    Sets the text, start and end attributes of the alignment files corresponding to the sampled features

    Parameters
    ----------
    data : Features Class
        Object containing all information to find alignments for the selected features
    sample : list
        List of file paths to the sampled features
    """

    alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled features
    data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files

if __name__ == "__main__":
    def plot_seg(segment, data, peaks, sample):
        """
        Plots the distances, moving average, hypothesized peaks and ground truth alignments for the first file in the sample 

        Parameters
        ----------
        segment : Segmentor Class
            Object containing all hyperparameters and methods to segment the features into words
        data : Features Class
            Object containing all information to find alignments for the selected features
        peaks : list (int)
            The frame indices of the detected peaks
        """

        # Plot Mel spectrogram:
        f_s = 16000
        n_fft = int(0.02*f_s) #20ms window length
        stride = int(0.0201*f_s) #20ms to result in same number of frames as the speech models
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=f_s, n_fft=n_fft, hop_length=stride, n_mels=64)

        paper_utt_idx = 206 # LibriSpeech Utterance 3853/163249/3853-163249-0040
        sample = sample[paper_utt_idx]
        sample = os.path.split(sample)[-1]
        file_path = glob(os.path.join("/media/hdd/data/librispeech/dev-clean", "**" , Path(sample).with_suffix(".flac")), recursive=True)[0]
        wav, _ = torchaudio.load(file_path, backend='soundfile')
        mel_specgram = np.log(transform(wav))
        # mel_specgram = mel_specgram.permute(0, 2, 1)
        if torch.isinf(mel_specgram).any():
            mel_specgram[torch.isinf(mel_specgram)] = torch.nan

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(mel_specgram.squeeze(), origin="lower", interpolation="nearest", zorder=0)

        # Plot the distances, moving average and peaks
        peaks = peaks[paper_utt_idx]
        distances = segment.distances[paper_utt_idx]*13 + (13)
        smoothed_distances = segment.smoothed_distances[paper_utt_idx]*13 + (13)
        
        # _, ax = plt.subplots()
        ax.plot(distances, label='Distances', color='red', alpha=0.7, linewidth=1.8, zorder=1)
        # ax.plot(np.range(segment.distances[paper_utt_idx]), segment.distances[paper_utt_idx] label='Distances', color='blue')
        ax.plot(smoothed_distances, label='Smooth Distances', color='white', alpha=1.0, linewidth=2.45, zorder=2)
        ax.scatter(peaks, smoothed_distances[peaks], marker='+', label='Peaks', color='black', s=200, linewidth=2.45, zorder=3)
        ax.scatter(peaks, smoothed_distances[peaks], marker='+', label='Peaks', color='white', s=150, linewidth=2.15, zorder=4)

        ax.fill_between([-5, len(smoothed_distances)], [-0.5,-200], color='white')

        alignment_end_frames = data.alignment_data[paper_utt_idx].end
        alignment_end_times = [data.get_sample_second(end_frame) for end_frame in alignment_end_frames]
        segments = [(a + ((b-a)//2)) for a, b in itertools.pairwise(np.concatenate(([0],alignment_end_frames)))]
        print('Segment end times, frames, and text')
        print(alignment_end_times)
        print(alignment_end_frames)
        print(segments)
        print(data.alignment_data[paper_utt_idx].text)

        ax.set_ylim(ymin=-10)
        for frame in alignment_end_frames:
            ax.axvline(x=frame, label='Ground Truth', color='black', linewidth=0.8)
            # ax.axvline(x=frame-data.get_frame_num(0.02), color='black', linewidth=0.2, alpha=0.5) # 0.02s tolerance
            # ax.axvline(x=frame+data.get_frame_num(0.02), color='black', linewidth=0.2, alpha=0.5)
        
        custom_tick_labels = data.alignment_data[paper_utt_idx].text
        ax.set_xticks([])
        ax.set_yticks([])
        font_size = 15/alignment_end_frames[-1] * 140
        y_lim = 150/alignment_end_frames[-1] * - 6
        # y_lim = (font_size * -0.8) - (font_size*0.025)
        for i, midpoint in enumerate(segments):
            ax.text(midpoint, -1, custom_tick_labels[i], ha='center', va='top', fontsize=font_size)

        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlim(xmin=alignment_end_frames[0]-5)
        ax.set_xlim(xmax=alignment_end_frames[-2]+5)
        ax.set_ylim(ymin=y_lim, ymax=63)
        plt.tight_layout()
        plt.savefig('distances.pdf', bbox_inches='tight')
  
    parser = argparse.ArgumentParser(description="Segment speech audio.")
    parser.add_argument(
        "model",
        help="the model used for feature extraction",
        default="mfcc",
    )
    parser.add_argument(
        "layer", # -1 for no layer
        type=int,
    )
    parser.add_argument(
        "features_dir",
        metavar="features-dir",
        help="path to the features directory.",
        type=Path,
    )
    parser.add_argument(
        "sample_size",
        metavar="sample-size",
        help="number of features to sample (-1 to sample all available data).",
        type=int,
    )
    parser.add_argument(
        "--save_out",
        help="option to save the output segment boundaries in ms to the specified directory.",
        default=None,
        type=Path,
    )
    parser.add_argument( # optional argument to load the optimized hyperparameters from a file
        '--load_hyperparams',
        default=None,
        type=Path,
    )
    parser.add_argument(
        '--batch_size',
        default=25000,
        type=int,
    )

    args = parser.parse_args()

    if args.load_hyperparams is None: # ask for hyperparameters
        print("Enter the hyperparameters for the segmentation algorithm: ")
        dist = str(input("Distance metric (euclidean, cosine): "))
        window = int(input("Moving average window size (int): "))
        prom = float(input("Peak detection prominence value (float): "))
    else:
        with open(args.load_hyperparams) as json_file:
            params = json.load(json_file)
            dataset_name = args.alignments_dir.stem
            if args.layer != -1:
                params = params[args.model][str(args.layer)]
            else:
                params = params[args.model]
            dist = params['distance']
            window = params['window_size']
            prom = params['prominence']
    
    if args.model in ["mfcc", "melspec"]:
        frames_per_ms = 10
    else:
        frames_per_ms = 20
    data = utils.Features(root_dir=args.features_dir, model_name=args.model, layer=args.layer, num_files=args.sample_size, frames_per_ms=frames_per_ms)

    # Features
    batch_num = 0
    batch = True

    while batch == True: # process in batches to avoid memory issues
        print("Batch number: ", batch_num)
        sample, features, norm_features, index_one_frame, batch, batch_num = get_features(data, args.batch_size, batch_num)

        # Remove features with only one frame
        sample_one_frame = [sample[i] for i in index_one_frame]
        for i in sorted(index_one_frame, reverse=True):
            del sample[i]
            del features[i]
            del norm_features[i]
        
        # Segmenting
        peaks, prominences, segmentor = get_word_segments(norm_features, distance_type=dist, prominence=prom, window_size=window)

        # Ensure last frame boundaries (except if last boundary is within tolerance of last frame)
        for i, peak in enumerate(peaks):
            if len(peak) == 0:
                peak = np.array([features[i].shape[0] - 1]) # add a peak at the end of the file
            elif peak[-1] != features[i].shape[0] and (peak[-1] != features[i].shape[0] - 1 or (frames_per_ms == 10 and peak[-1] != features[i].shape[0] - 2)): # add at last frame (if not there or in tolerance)
                peak = np.append(peak, features[i].shape[0] - 1)
            peaks[i] = peak

        # Add samples and peaks for features with only one frame
        sample.extend(sample_one_frame)
        peaks.extend([np.array([1]) for _ in range(len(sample_one_frame))])

        # Optionally save the output segment boundaries
        if args.save_out is not None:
            root_save = args.save_out / args.model / str(args.layer)
            for peak, file in tqdm(zip(peaks, sample), desc="Saving boundaries"):
                peak = data.get_sample_second(peak) # get peak to seconds
                save_dir = (root_save / os.path.split(file)[-1]).with_suffix(".list")
                save_dir.parent.mkdir(parents=True, exist_ok=True)
                with open(save_dir, "w") as f: # save the landmarks to a file
                    for l in peak:
                        f.write(f"{l}\n")

        del sample, features, norm_features, peaks, prominences, segmentor # clear memory for next batch