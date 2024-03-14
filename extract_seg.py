"""
Main script to extract embeddings, segment, and evaluate the audio data.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

from wordseg import utils, segment, evaluate
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def get_embeddings(data):
    sample = data.sample_embeddings() # sample from the feature embeddings
    embeddings = data.load_embeddings(sample) # load the sampled embeddings
    norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

    return sample, embeddings, norm_embeddings

def get_word_segments(norm_embeddings, distance_type="euclidean", prominence=0.6, window_size=5):
    # get the distances between adjacent frames in the embeddings
    segmentor = segment.Segmentor(distance_type=distance_type, prominence=prominence, window_size=window_size)
    segmentor.get_distance(norm_embeddings)
    print(norm_embeddings[0].shape)
    print(segmentor.distances[0].shape)

    # get the moving average of the distances
    segmentor.moving_average()
    print(segmentor.smoothed_distances[0].shape)

    # get the peaks in the distances
    peaks, prominences = segmentor.peak_detection()
    return peaks, prominences, segmentor

def set_alignments(data, sample):
    alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
    # print(alignments)
    data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files
    # print(data.alignment_data[0])

if __name__ == "__main__":
    def plot_seg(segment, data, peaks):
        fig, ax = plt.subplots()
        ax.plot(segment.distances[0], label='Distances', color='blue', alpha=0.5)
        # ax.plot(np.range(segment.distances[0]), segment.distances[0] label='Distances', color='blue')
        ax.plot(segment.smoothed_distances[0], label='Smooth Distances', color='red', alpha=0.5)
        ax.scatter(peaks, segment.smoothed_distances[0][peaks], marker='x', label='Peaks', color='green')
        
        alignment_end_times = data.alignment_data[0].end
        alignment_end_frames = [data.get_frame_num(end_time) for end_time in alignment_end_times]
        print(alignment_end_times)
        print(alignment_end_frames)

        for frame in alignment_end_frames:
            ax.axvline(x=frame, label='Ground Truth', color='black', linewidth=0.5)
        
        custom_ticks = alignment_end_frames
        custom_tick_labels = data.alignment_data[0].text
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_tick_labels, rotation=90, fontsize=6)

        plt.savefig('distances.png', dpi=300)
        
    parser = argparse.ArgumentParser(description="Segment speech audio.")
    parser.add_argument(
        "model",
        help="available models (wav2vec2.0: HuggingFace, fairseq, hubert: HuggingFace, fairseq, hubert:main)",
        choices=["w2v2_hf", "w2v2_fs", "hubert_hf", "hubert_fs", "hubert_shall"],
        default="w2v2_hf",
    )
    parser.add_argument(
        "layer",
        help="available models (wav2vec2.0: HuggingFace, fairseq, hubert: HuggingFace, fairseq, hubert:main)",
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
    parser.add_argument( # optional argument to load the optimized hyperparameters from a file
        '--load_hyperparams',
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args() #python3 extract_seg.py w2v2_hf 12 /media/hdd/embeddings /media/hdd/data/librispeech_alignments 5 --load_hyperparams
    
    if not args.load_hyperparams: # ask for hyperparameters
        print("Enter the hyperparameters for the segmentation algorithm: ")
        dist = input("Distance metric (euclidean, cosine): ")
        window = input("Moving average window size (int): ")
        prom = input("Peak detection prominence value (float): ")
    else: #load from file
        with open('optimized_parameters.json') as json_file:
            data = json.load(json_file)
            data = data[args.model][str(args.layer)]
            dist = data['distance']
            window = data['window_size']
            prom = data['prominence']
            print(data, dist, window, prom)
    
    data = utils.Features(root_dir=args.embeddings_dir, model_name=args.model, layer=args.layer, data_dir=args.alignments_dir, num_files=args.sample_size)

    # Embeddings
    sample, embeddings, norm_embeddings = get_embeddings(data)
    
    # Segmenting TODO write function to extract the optimised hyperparameters from the saved file
    peaks, prominences, segmentor = get_word_segments(norm_embeddings, distance_type=dist, prominence=prom, window_size=window)

    # Alignments
    set_alignments(data, sample)

    # Plot the distances, moving average and peaks AND compare to the alignments TODO make work for more than one sample
    plot_seg(segmentor, data, peaks)

    # Evaluate
    alignment_end_times = [alignment.end for alignment in data.alignment_data[:]]
    alignment_end_frames = []
    for alignment_times in alignment_end_times:
        alignment_frames = [data.get_frame_num(end_time) for end_time in alignment_times]
        alignment_end_frames.append(alignment_frames)
        
    p, r, f = evaluate.eval_segmentation(peaks, alignment_end_frames)
    rval = evaluate.get_rvalue(p, r)
    print(p, r, f, rval)