"""
Main script to extract embeddings, segment the audio, and evaluate the resulting segmentation.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

from wordseg import utils, segment, evaluate
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

def get_embeddings(data, batch_num):
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
    if 25000*(batch_num+1) <= len(sample):
        sample = sample[25000*batch_num:25000*(batch_num+1)]
        batch = True
    else:
        sample = sample[25000*batch_num:]
        batch = False
    batch_num = batch_num + 1

    embeddings = data.load_embeddings(sample) # load the sampled embeddings
    norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

    index_del = []
    for i, embedding in enumerate(norm_embeddings): # delete embeddings with only one frame
        if embedding.shape[0] == 1:
            index_del.append(i)

    # for i in sorted(index_del, reverse=True):
    #     del sample[i]
    #     del embeddings[i]
    #     del norm_embeddings[i]
    
    if len(sample) == 0:
        print('No embeddings to segment, sampled a file with only one frame.')
        exit()
    
    return sample, embeddings, norm_embeddings, index_del, batch, batch_num

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
    def plot_seg(segment, data, peaks, sample):
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

        # Plot Mel spectrogram:
        f_s = 16000
        n_fft = int(0.02*f_s) #20ms window length
        stride = int(0.0201*f_s) #20ms to result in same number of frames as the speech models
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=f_s, n_fft=n_fft, hop_length=stride, n_mels=64)

        sample = sample[0]
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
        peaks = peaks[0]
        distances = segment.distances[0]*13 + (13)
        smoothed_distances = segment.smoothed_distances[0]*13 + (13)
        
        # _, ax = plt.subplots()
        ax.plot(distances, label='Distances', color='red', alpha=0.7, linewidth=1.8, zorder=1)
        # ax.plot(np.range(segment.distances[0]), segment.distances[0] label='Distances', color='blue')
        ax.plot(smoothed_distances, label='Smooth Distances', color='white', alpha=1.0, linewidth=2.45, zorder=2)
        ax.scatter(peaks, smoothed_distances[peaks], marker='+', label='Peaks', color='black', s=200, linewidth=2.45, zorder=3)
        ax.scatter(peaks, smoothed_distances[peaks], marker='+', label='Peaks', color='white', s=150, linewidth=2.15, zorder=4)

        ax.fill_between([-5, len(smoothed_distances)], [-0.5,-200], color='white')

        alignment_end_frames = data.alignment_data[0].end
        alignment_end_times = [data.get_sample_second(end_frame) for end_frame in alignment_end_frames]
        segments = [(a + ((b-a)//2)) for a, b in itertools.pairwise(np.concatenate(([0],alignment_end_frames)))]
        print('Segment end times, frames, and text')
        print(alignment_end_times)
        print(alignment_end_frames)
        print(segments)
        print(data.alignment_data[0].text)

        ax.set_ylim(ymin=-10)
        for frame in alignment_end_frames:
            ax.axvline(x=frame, label='Ground Truth', color='black', linewidth=0.8)
            # ax.axvline(x=frame-data.get_frame_num(0.02), color='black', linewidth=0.2, alpha=0.5) # 0.02s tolerance
            # ax.axvline(x=frame+data.get_frame_num(0.02), color='black', linewidth=0.2, alpha=0.5)
        
        custom_tick_labels = data.alignment_data[0].text
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
        help="available models (wav2vec2.0: HuggingFace, fairseq, hubert: HuggingFace, fairseq, hubert:main)",
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

    args = parser.parse_args() # python3 extract_seg.py hubert_shall 10 /media/hdd/embeddings/librispeech /media/hdd/data/librispeech_alignments -1 --load_hyperparams --save_out=/media/hdd/segments/tti_wordseg/librispeech/dev_clean --strict
    # python3 extract_seg.py w2v2_hf 12 /media/hdd/embeddings/buckeye/dev /media/hdd/data/buckeye_alignments/dev -1 --align_format=.txt --load_hyperparams --strict
    # python3 extract_seg.py hubert_shall 10 /media/hdd/embeddings/buckeye/test /media/hdd/data/buckeye_alignments/test -1 --align_format=.txt --save_out=/media/hdd/segments/tti_wordseg/buckeye/test --load_hyperparams --strict
    # python3 extract_seg.py hubert_shall 10 /media/hdd/embeddings/zrc/zrc2017_train_segments/english /media/hdd/data/zrc_alignments/zrc2017_train_alignments/english -1 --align_format=.txt --save_out=/media/hdd/segments/tti_wordseg/zrc2017_train_segments/english --load_hyperparams --strict

    if not args.load_hyperparams: # ask for hyperparameters
        print("Enter the hyperparameters for the segmentation algorithm: ")
        dist = str(input("Distance metric (euclidean, cosine): "))
        window = int(input("Moving average window size (int): "))
        prom = float(input("Peak detection prominence value (float): "))
    else:
        with open('optimized_parameters.json') as json_file:
            params = json.load(json_file)
            dataset_name = args.alignments_dir.stem
            if args.layer != -1:
                params = params[args.model][str(args.layer)]
            else:
                params = params[args.model]
    
    if args.model in ["mfcc", "melspec"]:
        frames_per_ms = 10
    else:
        frames_per_ms = 20
    data = utils.Features(root_dir=args.embeddings_dir, model_name=args.model, layer=args.layer, data_dir=args.alignments_dir, alignment_format=args.align_format, num_files=args.sample_size, frames_per_ms=frames_per_ms)

    # Embeddings
    batch_num = 0
    batch = True

    while batch == True: # process in batches to avoid memory issues
        print("Batch number: ", batch_num)
        sample, embeddings, norm_embeddings, index_one_frame, batch, batch_num = get_embeddings(data, batch_num)

        # Remove embeddings with only one frame
        sample_one_frame = [sample[i] for i in index_one_frame]
        for i in sorted(index_one_frame, reverse=True):
            del sample[i]
            del embeddings[i]
            del norm_embeddings[i]
        
        # Segmenting
        peaks, prominences, segmentor = get_word_segments(norm_embeddings, distance_type=dist, prominence=prom, window_size=window)

        # Add samples and peaks for embeddings with only one frame
        sample.extend(sample_one_frame)
        peaks.extend([np.array([1]) for _ in range(len(sample_one_frame))])

        # Ensure last frame boundaries (except if last boundary is within tolerance of last frame)
        for i, peak in enumerate(peaks):
            if len(peak) == 0: # -1 to compensate for padding
                peak = np.array([embeddings[i].shape[0] - 1]) # add a peak at the end of the file
            elif i < len(embeddings): # ensure there is a peak at the last frame (at samples longer than one frame)
                # if abs(peak[-1] - embeddings[i].shape[0]) < 2: # in tolerance: remove and add at last frame
                #     peak[-1] = embeddings[i].shape[0] - 1
                if peak[-1] != embeddings[i].shape[0] and peak[-1] != embeddings[i].shape[0] - 1: # not in tolerance: add at last frame
                    peak = np.append(peak, embeddings[i].shape[0] - 1)
            peaks[i] = peak

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

        # Alignments
        align_sample = data.get_alignment_paths(sample)
        data.set_alignments(align_sample)

        # Evaluate
        alignment_end_frames = [alignment.end for alignment in data.alignment_data[:]]

        p, r, f = evaluate.eval_segmentation(peaks, alignment_end_frames, strict=args.strict)
        rval = evaluate.get_rvalue(p, r)
        print('Evaluation: \n Precision: ', p, '\n Recall: ', r, '\n F1-Score: ', f, '\n R-Value: ', rval)
        
        # Plot the distances, moving average and peaks AND compare to the alignments of a single file
        plot_seg(segmentor, data, peaks, sample)

        del align_sample, alignment_end_frames

        del sample, embeddings, norm_embeddings, peaks, prominences, segmentor # clear memory for next batch