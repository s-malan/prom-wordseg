"""
Funtions used to apply word segementation on sampled embeddings.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from sklearn.preprocessing import StandardScaler

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
        
    def get_distance(self, embeddings):
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
        
        for embedding in tqdm(embeddings, desc="Calculating Distances"):
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

    def moving_average(self):
        """
        Calculates the moving average of the distances

        Parameters
        ----------
        self : Segmentor Class
            Object containing all hyperparameters and methods to segment the embeddings into words
        """

        for dist in tqdm(self.distances, desc="Moving Average"):
            dist = np.pad(dist, (self.window_size // 2, self.window_size // 2), mode='edge')
            box = np.ones(self.window_size) / self.window_size
            self.smoothed_distances.append(np.convolve(dist, box, 'valid'))

    def peak_detection(self):
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

        for smooth_distance in tqdm(self.smoothed_distances, desc="Peak Detection"):
            peaks_found, _ = find_peaks(smooth_distance, prominence=self.prominence)
            prominences_found = peak_prominences(smooth_distance, peaks_found)[0]
            peaks.append(peaks_found)
            prominences.append(prominences_found)
        
        return peaks, prominences