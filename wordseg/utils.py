"""
Utility functions to sample audio features, normalize them, and get the corresponding alignments with their attributes.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: February 2024
"""

import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler

class Features:
    """
    The object containing all information to find alignments for the selected features.

    Parameters
    ----------
    root_dir : String
        The path to the root directory of the features
    model_name : String
        The name of the model to get the features from
    layer : int
        The number of the layer to get the features from
    num_files : int
        The number of features (utterances) to sample
        default: 2000
    frames_per_ms : int
        The number of frames a model processes per millisecond
        default: 20
    """

    def __init__(
        self, root_dir, model_name, layer, num_files=2000, frames_per_ms=20
    ):
        self.root_dir = root_dir
        self.model_name = model_name
        self.layer = layer
        self.num_files = num_files
        self.frames_per_ms = frames_per_ms

    def sample_features(self):
        """
        Randomly samples features from the specified model and returns the file paths as a list.

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected features

        Return
        ------
        features_sample : list
            List of file paths to the sampled features
        """

        if self.layer != -1:
            layer = 'layer_' + str(self.layer)
            all_features = glob(os.path.join(self.root_dir, self.model_name, layer, "**/*.npy"), recursive=True)
        else:
            all_features = glob(os.path.join(self.root_dir, self.model_name, "**/*.npy"), recursive=True)

        if self.num_files == -1: # sample all the data
            return all_features
        
        features_sample = np.random.choice(all_features, self.num_files, replace=False)
        return features_sample

    def load_features(self, files):
        """
        Load the sampled features from file paths

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected features
        files : list (String)
            List of file paths to the sampled features

        Return
        ------
        features : list
            A list of features loaded from the file paths
        """

        features = []

        for file in tqdm(files, desc="Loading features"):
            encodings = torch.from_numpy(np.load(file))
            if len(encodings.shape) == 1: # if only one dimension, add a dimension
                features.append(encodings.unsqueeze(0))
            else:
                features.append(encodings)
        return features
    
    def normalize_features(self, features):
        """
        Normalizes the features to have a mean of 0 and a standard deviation of 1

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected features
        features : numpy.ndarray
            The features to normalize

        Returns
        -------
        normalized_features : numpy.ndarray
            The normalized features
        """

        stacked_features = torch.cat(features, dim=0) # concatenate all features into one tensor with size (sum_seq_len, feature_dim (channels))

        scaler = StandardScaler()
        scaler.partial_fit(stacked_features) # (n_samples, n_features)
        normalized_features = []
        for feature in tqdm(features, desc="Normalizing Features"):
            normalized_features.append(torch.from_numpy(scaler.transform(feature))) # (n_samples, n_features)
        return normalized_features

    def get_frame_num(self, seconds):
        """
        Convert seconds to feature frame number

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected features
        seconds : float
            The number of seconds (of audio) to convert to frames

        Return
        ------
        output : int
            The feature frame number corresponding to the given number of seconds 
        """

        return np.round(seconds / self.frames_per_ms * 1000) # seconds (= samples / sample_rate) / 20ms per frame * 1000ms per second

    def get_sample_second(self, frame_num):
        """
        Convert feature frame number to seconds

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected features
        frame_num : float
            The frame number (of features) to convert to seconds

        Return
        ------
        output : double
            The number of seconds corresponding to the given feature frame number
        """

        return frame_num * self.frames_per_ms / 1000 # frame_num * 20ms per frame / 1000ms per second