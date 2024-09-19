"""
Utility functions to sample audio embeddings, normalize them, and get the corresponding alignments with their attributes.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: February 2024
"""

import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import os
from pathlib import Path
import textgrids # https://pypi.org/project/praat-textgrids/
from sklearn.preprocessing import StandardScaler # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

class Alignment_Data:
    """
    The object containing all information of a specific alignment file

    Parameters
    ----------
    dir : String
        The path to the alignment data
    alignment_format : String
        Format of the alignment file ('TextGrid', 'txt')
    type : String
        The type of alignment to extract from the file (e.g. 'words', 'phones')
        default: 'words'

    Attributes
    ----------
    text : list (String)
        The text of the alignment
    start : list (int)
        The start frame of the alignment
    end : list (int)
        The end frame of the alignment
    """
    def __init__(
        self, dir, alignment_format, type='words'
    ):
        self.dir = dir
        self.alignment_format = alignment_format
        self.type = type
        self.text = []
        self.layer = []
        self.start = []
        self.end = []

    def set_attributes(self, features):
        """
        Sets the text, start and end attributes of the object from the alignment file

        Parameters
        ----------
        self : Class
            The object containing all information of a specific alignment file
        """

        if self.alignment_format == '.TextGrid':
            for word in textgrids.TextGrid(self.dir)[self.type]:
                self.text.append(word.text)
                self.start.append(float(word.xmin))
                self.end.append(float(word.xmax))
        elif self.alignment_format == '.txt':
            with open(self.dir, 'r') as f:
                for line in f:
                    line = line.split()
                    self.start.append(float(line[0]))
                    self.end.append(float(line[1]))
                    self.text.append(line[2])
        
        self.start = features.get_frame_num(np.array(self.start)).astype(int)
        self.end = features.get_frame_num(np.array(self.end)).astype(int)
    
    def __str__(self):
        return f"Alignment_Data({self.dir}, {self.text}, {self.start}, {self.end})"

class Features:
    """
    The object containing all information to find alignments for the selected embeddings.

    Parameters
    ----------
    root_dir : String
        The path to the root directory of the embeddings
    model_name : String
        The name of the model to get the embeddings from
    layer : int
        The number of the layer to get the embeddings from
    data_dir : String
        The path to the root directory of the alignments
    alignment_format : String
        Format of the alignment files ('.TextGrid', '.txt')
        default: '.TextGrid'
    num_files : int
        The number of embeddings (utterances) to sample
        default: 2000
    frames_per_ms : int
        The number of frames a model processes per millisecond
        default: 20
    alignment_data : list (Alignment_Data)
        The objects containing all information of a specific alignment file
    """

    def __init__(
        self, root_dir, model_name, layer, data_dir, alignment_format='.TextGrid', num_files=2000, frames_per_ms=20
    ):
        self.root_dir = root_dir
        self.model_name = model_name
        self.layer = layer
        self.data_dir = data_dir
        self.alignment_format = alignment_format
        self.num_files = num_files
        self.frames_per_ms = frames_per_ms
        self.alignment_data = []

    def sample_embeddings(self): # Works
        """
        Randomly samples embeddings from the specified model and returns the file paths as a list.

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected embeddings

        Return
        ------
        embeddings_sample : list
            List of file paths to the sampled embeddings
        """

        if self.layer != -1:
            layer = 'layer_' + str(self.layer)
            all_embeddings = glob(os.path.join(self.root_dir, self.model_name, layer, "**/*.npy"), recursive=True)
        else:
            all_embeddings = glob(os.path.join(self.root_dir, self.model_name, "**/*.npy"), recursive=True)

        if self.num_files == -1: # sample all the data
            return all_embeddings
        
        embeddings_sample = np.random.choice(all_embeddings, self.num_files, replace=False)
        return embeddings_sample

    def load_embeddings(self, files): # Works
        """
        Load the sampled embeddings from file paths

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected embeddings
        files : list (String)
            List of file paths to the sampled embeddings

        Return
        ------
        embeddings : list
            A list of embeddings loaded from the file paths
        """

        embeddings = []

        for file in tqdm(files, desc="Loading Embeddings"):
            embedding = torch.from_numpy(np.load(file))
            if len(embedding.shape) == 1: # if only one dimension, add a dimension
                embeddings.append(embedding.unsqueeze(0))
            else:
                embeddings.append(embedding)
        return embeddings
    
    def normalize_features(self, features): # Works
        """
        Normalizes the feature embeddings to have a mean of 0 and a standard deviation of 1

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected embeddings
        features : numpy.ndarray
            The feature embeddings to normalize

        Returns
        -------
        normalized_features : numpy.ndarray
            The normalized feature embeddings
        """

        stacked_features = torch.cat(features, dim=0) # concatenate all features into one tensor with size (sum_seq_len, feature_dim (channels))

        scaler = StandardScaler()
        scaler.partial_fit(stacked_features) # (n_samples, n_features)
        normalized_features = []
        for feature in tqdm(features, desc="Normalizing Features"):
            normalized_features.append(torch.from_numpy(scaler.transform(feature))) # (n_samples, n_features)
        return normalized_features

    def get_alignment_paths(self, files): # Works
        """
        Find Paths to the TextGrid files (alignments) corresponding to the sampled embeddings

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected embeddings
        files : list (String)
            List of file paths to the sampled embeddings

        Return
        ------
        alignments : list (Path)
            A list of file paths to the alignment files corresponding to the sampled embeddings
        """

        alignments = []

        for file in tqdm(files, desc="Sample Alignments"): # TODO make this work better, the -4 is a hack
            if self.alignment_format == '.TextGrid':
                sub_dir = file.split("/")[-4:]
                align_dir = os.path.join(self.data_dir, *sub_dir)
            elif self.alignment_format == '.txt':
                sub_dir = file.split("/")[-1]
                align_dir = os.path.join(self.data_dir, sub_dir)
            
            if os.path.exists(Path(align_dir).with_suffix(self.alignment_format)):
                alignments.append(Path(align_dir).with_suffix(self.alignment_format))

        return alignments
    
    def set_alignments(self, files): # Works
        """
        Create Alignment_Data objects and set their attributes for each alignment file

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected embeddings
        files : list (Path)
            A list of file paths to the alignment files corresponding to the sampled embeddings
        """
            
        for file in tqdm(files, desc="Reading Alignment"):
            alignment = Alignment_Data(file, self.alignment_format, type='words')
            alignment.set_attributes(self)
            self.alignment_data.append(alignment)

    def get_frame_num(self, seconds): # Works
        """
        Convert seconds to feature embedding frame number

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected embeddings
        seconds : float
            The number of seconds (of audio) to convert to frames

        Return
        ------
        output : int
            The feature embedding frame number corresponding to the given number of seconds 
        """

        return np.round(seconds / self.frames_per_ms * 1000) # seconds (= samples / sample_rate) / 20ms per frame * 1000ms per second

    def get_sample_second(self, frame_num): # Works
        """
        Convert feature embedding frame number to seconds

        Parameters
        ----------
        self : Class
            The object containing all information to find alignments for the selected embeddings
        frame_num : float
            The frame number (of feature embeddings) to convert to seconds

        Return
        ------
        output : double
            The number of seconds corresponding to the given feature embedding frame number
        """

        return frame_num * self.frames_per_ms / 1000 # frame_num * 20ms per frame / 1000ms per second