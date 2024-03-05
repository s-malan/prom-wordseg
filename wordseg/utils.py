"""
Utility functions to sample audio encodings, ...

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: February 2024
"""

import numpy as np
import torch
import torchaudio

x = torch.from_numpy(np.load('embeddings/w2v2_fs/layer_12/librispeech/dev-clean/84/121123/84-121123-0000.npy'))

print(x)
print(x.shape)