"""
Funtions used to segment speech data into word units. Using peak detection on the distances between feature embeddings.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: February 2024
"""

import numpy as np
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.signal import peak_prominences # used to find the prominences of the peaks

def get_distance(embedding): # TODO check if works, also check axis
    """
    Calculates the distance between each embedding in the sequence

    Parameters
    ----------
    embeddings : numpy.ndarray
        The feature embeddings to calculate distances for

    Returns
    -------
    numpy.ndarray
        The distances between each embedding
    """
    embedding_dist = np.diff(embedding, axis=0)
    euclidean_dist = np.linalg.norm(embedding_dist, axis=1)

    cosine_distances = [distance.cosine(embedding[i], embedding[i + 1]) for i in range(embedding.shape[0] - 1)]

def moving_average(distance, window_size): # TODO check if works
    """
    Calculates the moving average of the distances

    Parameters
    ----------
    distance : numpy.ndarray
        The distances to calculate the moving average for
    window_size : int
        The size of the window to calculate the moving average over

    Returns
    -------
    numpy.ndarray
        The moving average of the distances
    """
    return np.convolve(distance, np.ones(window_size), 'valid') / window_size

def peak_detection(distances, prominence=0.1): # TODO check if works
    """
    Finds the peaks in the distances between feature embeddings

    Parameters
    ----------
    distance : numpy.ndarray
        The distances to find peaks in
    prominence : float
        The prominence of the peaks
    distance : int
        The minimum distance between peaks
    width : int
        The width of the peaks

    Returns
    -------
    numpy.ndarray
        The indices of the peaks
    """
    peaks, _ = find_peaks(distances, prominence=prominence) #, distance=distance, width=width)
    prominences = peak_prominences(distances, peaks)[0]
    return peaks, prominences

def grid_search(): # TODO grid search over: dist_type, moving_avg_window, prominence value threshold
    pass