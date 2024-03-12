"""
Encode audio data with different models to be used downstream for word segmentation.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: February 2024
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os

import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.functional import resample

#loading from fairseq or HuggingFace
import fairseq
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import HubertModel, AutoProcessor


class EncodeAudio:
    def __init__(
        self, model_name, data_dir, save_dir, extension #TODO add dataset argument (default librispeech)
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.extension = extension

        # load model for future use
        if self.model_name == "w2v2_hf":
            # self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        elif self.model_name == "w2v2_fs":
            ckpt_path = '/media/hdd/models/wav2vec_small.pt'
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
            self.model = models[0]
            self.model.eval()
        elif self.model_name == "hubert_hf":
            # self.processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
            self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        elif self.model_name == "hubert_fs":
            ckpt_path = "/media/hdd/models/hubert_base_ls960.pt"
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path], strict=False)
            self.model = models[0]
            self.model.eval()
        else: #hubert_shall
            self.model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda()        
    
    def save_embedding(self, wav, file_path):
        """
        Push the audio through the model and save the embeddings to the save directory.
        Do for each layer of the model
        """
        
        if self.model_name == "w2v2_hf":
            self.encode_w2v2_hf(wav, file_path)
        elif self.model_name == "w2v2_fs":
            self.encode_w2v2_fairseq(wav, file_path)
        elif self.model_name == "hubert_hf":
            self.encode_hubert_hf(wav, file_path)
        elif self.model_name == "hubert_fs":
            self.encode_hubert_fairseq(wav, file_path)
        elif self.model_name == "hubert_shall": 
            self.encode_hubert_shall(wav, file_path)
    
    def encode_w2v2_hf(self, wav, file_path): # Works
        """
        Determines the embeddings of the audio using the wav2vec2 model from HuggingFace.
        Saves the embeddings to the save directory using the same structure as the dataset directory.

        Parameters
        ----------
        self : encoder object
            The model type, data directory, save directory, and extension.
        wav : tensor
            The audio waveform
        sr : int
            The sample rate of the audio
        file_path : String
            The path to the audio file

        Return
        ------
        output : N/A
        """

        layer = 12  # Change this to the desired layer index [0,12]

        # Preprocess the input audio
        # input_values = self.processor(wav, return_tensors="pt", padding="longest").input_values

        # Forward pass through the modified model
        with torch.no_grad():
            x = self.model.forward(wav, output_hidden_states=True, output_attentions=False)

        x_layers = x[2][1:] # extract features for each layer

        for i in range(len(x_layers)):
            x = x_layers[i]
            
            # out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / file_path.relative_to(self.data_dir.parts[0])
            path_prefix = os.path.commonprefix([self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}'), file_path])
            out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / os.path.relpath(file_path, path_prefix)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())

    def encode_w2v2_fairseq(self, wav, file_path): # Works
        """
        Determines the embeddings of the audio using the wav2vec2 model from fairseq.
        Saves the embeddings to the save directory using the same structure as the dataset directory.

        Parameters
        ----------
        self : encoder object
            The model type, data directory, save directory, and extension.
        wav : tensor
            The audio waveform
        sr : int
            The sample rate of the audio
        file_path : String
            The path to the audio file

        Return
        ------
        output : N/A
        """

        layer = 12  # Replace this with the desired layer number

        with torch.no_grad(): #https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py#L795
            x = self.model.extract_features(wav, layer = layer, padding_mask = None)
        
        # extract features for each layer
        # x = x['x'] # or get each x from for loop through x['layer_results'] after layer = 12
        x_layers = x['layer_results']
        
        # print(x.keys())
        # print(x_layers[layer][0])
        # print(x['x'])
        # print('~~~ x ~~~', x['x'], '\n ~~~ pm ~~~', x['padding_mask'], '\n ~~~ feat ~~~', x['features'], '\n ~~~ layer ~~~', x['layer_results'])
        # print(len(x['layer_results']))

        for i in range(len(x_layers)): # save each layer
            x = x_layers[i][0].permute(1, 0, 2)
            
            # out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / file_path.relative_to(self.data_dir.parts[0])
            path_prefix = os.path.commonprefix([self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}'), file_path])
            out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / os.path.relpath(file_path, path_prefix)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())

    def encode_hubert_hf(self, wav, file_path): # Works
        """
        Determines the embeddings of the audio using the HuBERT model from HuggingFace.
        Saves the embeddings to the save directory using the same structure as the dataset directory.

        Parameters
        ----------
        self : encoder object
            The model type, data directory, save directory, and extension.
        wav : tensor
            The audio waveform
        sr : int
            The sample rate of the audio
        file_path : String
            The path to the audio file

        Return
        ------
        output : N/A
        """

        # Choose the nth layer you want to extract features from
        layer = 12  # Replace this with the desired layer number

        # Forward pass through the model
        with torch.no_grad():
            x = self.model.forward(wav, output_hidden_states=True, output_attentions=False)

        x_layers = x[1] #all 12 layer features

        for i in range(len(x_layers)):
            x = x_layers[i]
            
            # out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / file_path.relative_to(self.data_dir.parts[0])
            path_prefix = os.path.commonprefix([self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}'), file_path])
            out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / os.path.relpath(file_path, path_prefix)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())

    def encode_hubert_fairseq(self, wav, file_path): # Works
        """
        Determines the embeddings of the audio using the HuBERT model from fairseq.
        Saves the embeddings to the save directory using the same structure as the dataset directory.

        Parameters
        ----------
        self : encoder object
            The model type, data directory, save directory, and extension.
        wav : tensor
            The audio waveform
        sr : int
            The sample rate of the audio
        file_path : String
            The path to the audio file

        Return
        ------
        output : N/A
        """

        layer = 12  # Replace this with the desired layer number

        for i in range(layer):
            with torch.no_grad(): #https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/hubert/hubert.py
                x, _ = self.model.extract_features(wav, output_layer = i)

            # out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / file_path.relative_to(self.data_dir.parts[0])
            path_prefix = os.path.commonprefix([self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}'), file_path])
            out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / os.path.relpath(file_path, path_prefix)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())
    
    def encode_hubert_shall(self, wav, file_path): # Works
        """
        Determines the embeddings of the audio using the HuBERT model from fairseq using a wrapper function.
        Saves the embeddings to the save directory using the same structure as the dataset directory.

        Parameters
        ----------
        self : encoder object
            The model type, data directory, save directory, and extension.
        wav : tensor
            The audio waveform
        sr : int
            The sample rate of the audio
        file_path : String
            The path to the audio file

        Return
        ------
        output : N/A
        """

        wav = wav.unsqueeze(0).cuda()
        layer = 12

        for i in range(layer):
            with torch.inference_mode():
                wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
                x, _ = self.model.encode(wav, layer=i)
            
            # out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / file_path.relative_to(self.data_dir.parts[0])
            
            path_prefix = os.path.commonprefix([self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}'), file_path])
            out_path = (self.save_dir.joinpath(f'{self.model_name}',f'layer_{i+1}')) / os.path.relpath(file_path, path_prefix)
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())
    
    def get_encodings(self):
        """
        Return the encodings for the dataset.
        """

        # walk through data directory
        for (dirpath, dirnames, filenames) in os.walk(self.data_dir):
            if not filenames: # no files in directory
                continue

            print('dirpath', dirpath)
            print('dirnames', dirnames)
            print('filenames', filenames)
            
            # not in root of dataset path
            if dirpath is not self.data_dir:
                sub_dir = dirpath.split("/")[-1]
                print('subdir', sub_dir)

                # walk through files in directory
                for file in filenames:
                    if not file.endswith(self.extension): # ensure only audio files are processed
                        continue

                    file_path = os.path.join(dirpath, file)
                    print('filepath', file_path)

                    wav, sr = torchaudio.load(file_path, backend='soundfile')
                    assert sr == 16000

                    self.save_embedding(wav, Path(file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "model",
        help="available models (wav2vec2.0: HuggingFace, fairseq, hubert: HuggingFace, fairseq, hubert:main)",
        choices=["w2v2_hf", "w2v2_fs", "hubert_hf", "hubert_fs", "hubert_shall"],
        default="w2v2_hf",
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    args = parser.parse_args() #python3 wordseg/encode.py w2v2_hf data/librispeech/dev-clean/ embeddings/
                               #for hdd: python3 wordseg/encode.py w2v2_fs /media/hdd/data/librispeech/dev-clean/ /media/hdd/embeddings/

    # extract the audio from the dataset and save the embeddings
    encoder = EncodeAudio(args.model, args.in_dir, args.out_dir, args.extension)
    encoder.get_encodings()

    # fire.Fire(
    #     {
    #         "encode_w2v2_hf": encode_w2v2_hf,
    #         "encode_w2v2_fairseq": encode_w2v2_fairseq,
    #         "encode_hubert_hf": encode_hubert_hf,
    #         "encode_hubert_fairseq": encode_hubert_fairseq,
    #     }
    # )