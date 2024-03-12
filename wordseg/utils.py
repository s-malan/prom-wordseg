"""
Utility functions to sample audio embeddings, normalize them, and get the corresponding alignments with their attributes.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: February 2024
"""

import numpy as np
import torch
import torchaudio
from glob import glob
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
    type : String
        The type of alignment to extract from the file (e.g. 'words', 'phones')
        default: 'words'
    grid : TextGrid
        Contains the alignment data
    """
    def __init__(
        self, dir, type='words'
    ):
        self.dir = dir
        self.type = type
        self.grid = textgrids.TextGrid(self.dir)
        self.text = []
        self.layer = []
        self.start = []
        self.end = []

    def set_attributes(self):
        """
        Sets the text, start and end attributes of the object from the alignment file

        Parameters
        ----------
        self : Class
            The object containing all information of a specific alignment file
        """
        for word in self.grid[self.type]:
            self.text.append(word.text)
            self.start.append(word.xmin)
            self.end.append(word.xmax)
    
    def __str__(self):
        return f"Alignment_Data({self.dir}, {self.text}, {self.start}, {self.end})"

class Features:
    """
    The object containing all information to find alignments for the selected embeddings

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
        self, root_dir, model_name, layer, data_dir, num_files=2000, frames_per_ms=20
    ):
        self.root_dir = root_dir
        self.model_name = model_name
        self.layer = layer
        self.data_dir = data_dir
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

        layer = 'layer_' + str(self.layer) # TODO extract if not in subdirectory but just in the root_dir/model_name/layer directory
        all_embeddings = glob(os.path.join(self.root_dir, self.model_name, layer, "**/*.npy"), recursive=True)
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

        for file in files:
            embeddings.append(torch.from_numpy(np.load(file)))
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
        for feature in features:
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

        for file in files:
            sub_dir = file.split("/")[-4:]
            align_dir = os.path.join(self.data_dir, *sub_dir)
            
            if os.path.exists(Path(align_dir).with_suffix(".TextGrid")):
                alignments.append(Path(align_dir).with_suffix(".TextGrid"))

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
            
        for file in files:
            alignment = Alignment_Data(file, type='words')
            alignment.set_attributes()
            self.alignment_data.append(alignment)

    def get_frame_num(self, seconds): # Works
        """
        Find Paths to the TextGrid files (alignments) corresponding to the sampled embeddings

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

        return round(seconds / self.frames_per_ms * 1000) # seconds (= samples / sample_rate) / 20ms per frame * 1000ms per second

    def get_sample_second(self, frame_num): # Works
        """
        Find Paths to the TextGrid files (alignments) corresponding to the sampled embeddings

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

if __name__ == "__main__":
    x = torch.from_numpy(np.load('/media/hdd/embeddings/w2v2_fs/layer_1/librispeech/dev-clean/84/121123/84-121123-0000.npy'))
    print(x, x.shape)
    y = torchaudio.load('/media/hdd/data/librispeech/dev-clean/84/121123/84-121123-0000.flac')
    print(y[0].shape)

    # Example usage
    embeddings_dir = '/media/hdd/embeddings'
    model_name = 'w2v2_fs'
    layer = 12
    alignments_dir = '/media/hdd/data/librispeech_alignments'

    aligner = Features(root_dir=embeddings_dir, model_name=model_name, layer=layer, data_dir=alignments_dir, num_files=5)
    sample = aligner.sample_embeddings()
    print(sample)

    alignments = aligner.get_alignment_paths(files=sample)
    print(alignments)

    aligner.set_alignments(files=alignments)
    print(aligner.alignment_data[0])

    embeddings = aligner.load_embeddings(sample)
    # print(embeddings)
    # print(embeddings[0].shape)
    # print(embeddings[0][:,0].shape)

    print(torch.max(torch.mean(torch.cat(embeddings, dim=0), dim=0)))
    print(torch.max(torch.std(torch.cat(embeddings, dim=0), dim=0)))

    norm_embeddings = aligner.normalize_features(embeddings)
    # print(norm_embeddings[0])
    # print(norm_embeddings[0].shape)

    # check if the channels are normalized separately (get mean over each 768 channels for each sample)
    mean = torch.mean(torch.cat(norm_embeddings, dim=0), dim=0)
    std = torch.std(torch.cat(norm_embeddings, dim=0), axis=0)
    print(torch.max(mean))
    print(torch.max(std))