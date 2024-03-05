"""
Funtions used to segment speech data into word units. Using peak detection on the distances between feature embeddings.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: February 2024
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_prominences # used to find the prominences of the peaks

