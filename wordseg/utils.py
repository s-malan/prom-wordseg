"""
Utility functions to sample audio encodings, ...

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

class Alignments:
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
    dir_depth : int
        The number of subdirectories from the root directory/model_name/layer to the embeddings (.npy files)
        default: 5
    num_files : int
        The number of embeddings (utterances) to sample
        default: 2000
    frames_per_ms : int
        The number of frames a model processes per millisecond
        default: 20
    """
    def __init__(
        self, root_dir, model_name, layer, data_dir, dir_depth=5, num_files=2000, frames_per_ms=20
    ):
        self.root_dir = root_dir
        self.model_name = model_name
        self.layer = layer
        self.data_dir = data_dir
        self.dir_depth = dir_depth
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
        output : list
            List of file paths to the sampled embeddings
        """

        search_path = "/".join(["*"] * self.dir_depth)
        layer = 'layer_' + str(self.layer)
        all_embeddings = glob(os.path.join(self.root_dir, self.model_name, layer, search_path + ".npy"))
        embeddings_sample = np.random.choice(all_embeddings, self.num_files, replace=False)
        return embeddings_sample

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
        output : list (Path)
            A list of file paths to the alignment files corresponding to the sampled embeddings
        """

        alignments = []

        for file in files:
            sub_dir = file.split("/")[-4:]
            align_dir = os.path.join(self.data_dir, *sub_dir)
            
            if os.path.exists(Path(align_dir).with_suffix(".TextGrid")):
                alignments.append(Path(align_dir).with_suffix(".TextGrid"))

        return alignments
    
    def set_alignments(self, files):
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

        return int(seconds / self.frames_per_ms / 1000) # seconds (= samples / sample_rate) / 20ms per frame * 1000ms per second

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
        output : int
            The number of seconds corresponding to the given feature embedding frame number
        """

        return frame_num * self.frames_per_ms / 1000 # frame_num * 20ms per frame / 1000ms per second

if __name__ == "__main__":
    x = torch.from_numpy(np.load('embeddings/w2v2_fs/layer_1/librispeech/dev-clean/84/121123/84-121123-0000.npy'))
    print(x, x.shape)
    y = torchaudio.load('data/librispeech/dev-clean/84/121123/84-121123-0000.flac')
    print(y[0].shape)

    # Example usage
    embeddings_dir = 'embeddings'
    model_name = 'w2v2_fs'
    layer = 12
    alignments_dir = 'data/librispeech_alignments'

    aligner = Alignments(root_dir=embeddings_dir, model_name=model_name, layer=layer, data_dir=alignments_dir, num_files=1)
    sample = aligner.sample_embeddings()
    print(sample)

    alignments = aligner.get_alignment_paths(files=sample)
    print(alignments)

    aligner.set_alignments(files=alignments)
    print(aligner.alignment_data[0])