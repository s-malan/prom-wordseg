"""
Funtions used to apply word segementation on sampled embeddings. Also does grid search for hyperparameters when run as the main method. 

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

import numpy as np
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.signal import peak_prominences # used to find the prominences of the peaks
from sklearn.preprocessing import StandardScaler # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from evaluate import eval_segmentation

class Segmentor:
    """
    The object containing all hyperparameters and methods to segment the embeddings into words

    Parameters
    ----------
    distance_type : String
        The type of the distance metric used ('euclidean' or 'cosine')
    prominence : double
        Prominence value threshold for peak detection
    window_size : int
        The window size for the moving average (in number of frames (1 frame = 20ms for w2v2 and HuBERT))
    distances : list
        Distances between adjacent frames in the embeddings
    smoothed_distances : int
        The moving average of the distances
    """

    def __init__(
        self, distance_type, prominence=0.1, window_size=2
    ):
        self.distance = distance_type
        self.prominence = prominence
        self.window_size = window_size
        self.distances = []
        self.smoothed_distances = []
        
    def get_distance(self, embeddings): # Works
        """
        Calculates the normalized distance between each embedding in the sequence

        Parameters
        ----------
        self : Segmentor Class
            Object containing all hyperparameters and methods to segment the embeddings into words
        embeddings : numpy.ndarray
            The feature embeddings to calculate distances for
        """

        scaler = StandardScaler()
        
        for embedding in embeddings:
            if self.distance == "euclidean":
                embedding_dist = np.diff(embedding, axis=0)
                euclidean_dist = np.linalg.norm(embedding_dist, axis=1)
                scaler.fit(euclidean_dist.reshape(-1, 1))
                euclidean_dist = scaler.transform(euclidean_dist.reshape(-1, 1))
                self.distances.append(euclidean_dist.reshape(-1))
            elif self.distance == "cosine":
                cosine_distances = np.array([distance.cosine(embedding[i], embedding[i + 1]) for i in range(embedding.shape[0] - 1)])
                scaler.fit(cosine_distances.reshape(-1, 1))
                cosine_distances = scaler.transform(cosine_distances.reshape(-1, 1))
                self.distances.append(cosine_distances.reshape(-1))
            else:
                raise ValueError("Distance type not supported")

    def moving_average(self): # Works
        """
        Calculates the moving average of the distances

        Parameters
        ----------
        self : Segmentor Class
            Object containing all hyperparameters and methods to segment the embeddings into words
        """

        for dist in self.distances:
            dist = np.pad(dist, (self.window_size // 2, self.window_size // 2), mode='edge') # TODO check if this is the correct padding
            box = np.ones(self.window_size) / self.window_size
            self.smoothed_distances.append(np.convolve(dist, box, 'valid'))

    def peak_detection(self): # Works
        """
        Finds the peaks in the distances between feature embeddings

        Parameters
        ----------
        self : Segmentor Class
            Object containing all hyperparameters and methods to segment the embeddings into words

        Returns
        -------
        peaks : list (int)
            The frame indices of the detected peaks
        prominences : list (float)
            The prominence values of the detected peaks
        """

        peaks = []
        prominences = []

        for smooth_distance in self.smoothed_distances:
            peaks_found, _ = find_peaks(smooth_distance, prominence=self.prominence)
            prominences_found = peak_prominences(smooth_distance, peaks_found)[0]
            peaks.append(peaks_found)
            prominences.append(prominences_found)

        return peaks, prominences

    def grid_search(self, embeddings_dir, alignments_dir, sample_size=2000): # TODO grid search over: dist_type, moving_avg_window, prominence value threshold
        # TODO for each model
        model = 'w2v2_hf'
        # TODO for each layer
        layer = 12
        # TODO for each combination of hyperparameters:
        # distance {euclidean, cosine}
        # window_size {1, 2, 3, 4, 5}
        # prominence {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

        data = utils.Features(root_dir=embeddings_dir, model_name=model, layer=layer, data_dir=alignments_dir, num_files=sample_size)

        # Embeddings
        sample = data.sample_embeddings() # sample from the feature embeddings
        embeddings = data.load_embeddings(sample) # load the sampled embeddings
        norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

        # Segmenting
        segment = Segmentor(distance_type="cosine", prominence=0.65, window_size=2) # window_size is in frames (1 frame = 20ms for wav2vec2 and HuBERT)
        segment.get_distance(norm_embeddings) # calculate the normalized distance between each embedding in the sequence
        segment.moving_average() # calculate the moving average of the distances

        peaks, _ = segment.peak_detection() # find the peaks in the distances

        # Alignments
        alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
        data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files

        # Evaluate
        alignment_end_times = [alignment.end for alignment in data.alignment_data[:]]
        alignment_end_frames = []
        for alignment_times in alignment_end_times:
            alignment_frames = [data.get_frame_num(end_time) for end_time in alignment_times]
            alignment_end_frames.append(alignment_frames)
            
        p, r, f = eval_segmentation(peaks, alignment_end_frames)
        print(p, r, f)

        # TODO save the results for each all h-p's for a layer locally, then save the best results for h-p's for the layer (and model) to a file

if __name__ == "__main__":

    import utils
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
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
        help="number of embeddings to sample.",
        type=int,
    )

    args = parser.parse_args() #python3 wordseg/segment.py w2v2_hf 12 /media/hdd/embeddings /media/hdd/data/librispeech_alignments 5
    
    data = utils.Features(root_dir=args.embeddings_dir, model_name=args.model, layer=args.layer, data_dir=args.alignments_dir, num_files=args.sample_size)

    # Embeddings
    sample = data.sample_embeddings() # sample from the feature embeddings
    # print(sample)
    embeddings = data.load_embeddings(sample) # load the sampled embeddings
    norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings
    print('original embedding shape', norm_embeddings[0].shape)
    
    # Segmenting
    segment = Segmentor(distance_type="cosine", prominence=0.65, window_size=2) # window_size is in frames (1 frame = 20ms for wav2vec2 and HuBERT)
    segment.get_distance(norm_embeddings)
    print('distance shape', segment.distances[0].shape)

    segment.moving_average() # calculate the moving average of the distances
    print('smoothed distance shape', segment.smoothed_distances[0].shape)

    print('mean distance (normalized)', np.mean(segment.smoothed_distances[0]))
    print('std distance (normalized)', np.std(segment.smoothed_distances[0]))

    peaks, prominences = segment.peak_detection() # find the peaks in the distances
    print('peaks, prominences', peaks[0], prominences[0])

    # Alignments
    alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
    # print(alignments)
    data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files
    # print(data.alignment_data[0])

    # Plot the distances, moving average and peaks AND compare to the alignments
    fig, ax = plt.subplots()
    ax.plot(segment.distances[0], label='Distances', color='blue', alpha=0.5)
    # ax.plot(np.range(segment.distances[0]), segment.distances[0] label='Distances', color='blue')
    ax.plot(segment.smoothed_distances[0], label='Smooth Distances', color='red', alpha=0.5)
    ax.scatter(peaks, segment.smoothed_distances[0][peaks], marker='x', label='Peaks', color='green')
    
    alignment_end_times = data.alignment_data[0].end
    alignment_end_frames = [data.get_frame_num(end_time) for end_time in alignment_end_times]
    print('Alignment end times and frames:')
    print(alignment_end_times)
    print(alignment_end_frames)
    print(data.alignment_data[0].text)

    for frame in alignment_end_frames:
        ax.axvline(x=frame, label='Ground Truth', color='black', linewidth=0.5)
    
    custom_ticks = alignment_end_frames
    custom_tick_labels = data.alignment_data[0].text
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_tick_labels, rotation=90, fontsize=6)

    plt.savefig('distances.png', dpi=300)

    # TODO grid search for hyperparameters
    # https://scikit-learn.org/stable/modules/grid_search.html
    
    segment.grid_search(args.embeddings_dir, args.alignments_dir, args.sample_size)