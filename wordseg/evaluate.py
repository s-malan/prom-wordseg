"""
Funtions used to evaluate word segmentation algorithm.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

# TODO: NED, UWER

import numpy as np

def eval_segmentation(seg, ref, strict=True, tolerance=1):
    """
    Calculate precision, recall, F-score for the segmentation boundaries.

    Parameters
    ----------
    seg : list of list of int
        The segmentation hypothesis word boundary frames for all utterances in the sample.
    ref : list of list of int
        The ground truth reference word boundary frames for all utterances in the sample.
    tolerance : int
        The number of offset frames that a segmentation hypothesis boundary can have with regards to a reference boundary and still be regarded as correct.
        default: 1 (20ms)

    Return
    ------
    output : (float, float, float)
        precision, recall, F-score.
    """

    num_seg = 0 #Nf
    num_ref = 0 #Nref
    num_hit = 0 #Nhit
        
    assert len(seg) == len(ref) # Check if the number of utterances in the hypothesis and reference are the same
    for i_utterance in range(len(seg)):
        prediction = seg[i_utterance] # TODO remove last boundaty of prediction
        ground_truth = ref[i_utterance]
        ground_truth = ground_truth[:-1] # Remove the last boundary of the reference

        # HERMAN DOES
        # If lengths are the same, disregard last True reference boundary OF THE PREDICTED BOUNDARIES
        # if len(boundary_PRED) == len(boundary_GT):
        #     boundary_PRED = boundary_PRED[:-1] # PRED
        #     # boundary_GT = boundary_GT[:-1]
        
        # boundary_GT = seg[i_boundary][:-1]  # last boundary is always True, # GT
        #                                      # don't want to count this

        # # If reference is longer, truncate
        # if len(boundary_PRED) > len(boundary_GT):
        #     boundary_PRED = boundary_PRED[:len(boundary_GT)]
        # HERMAN ENDS

        num_seg += len(prediction)
        num_ref += len(ground_truth)

        # count the number of hits
        for i_ref in ground_truth: # TODO check, multiple ref can still hit on a single seg
            for i_seg in prediction:
                if abs(i_ref - i_seg) <= tolerance:
                    num_hit += 1
                    if strict: break # makes the evaluation strict, so that a reference boundary can only be hit once

    # Calculate metrics:
    precision = float(num_hit/num_seg)
    recall = float(num_hit/num_ref)
    if precision + recall != 0:
        f1_score = 2*precision*recall/(precision+recall)
    else:
        f1_score = -np.inf
    
    return precision, recall, f1_score

def get_os(precision, recall):
    """
    Calculates the over-segmentation; how many fewer/more boundaries are proposed compared to the ground truth.

    Parameters
    ----------
    precision : float
        How often word segmentation correctly predicts a word boundary.
    recall : float
        How often word segmentation's prediction matches a ground truth word boundary.

    Return
    ------
    output : float
        over-segmentation
    """

    if precision == 0:
        return -np.inf
    else:
        return recall/precision - 1
    
def get_rvalue(precision, recall):
    """
    Calculates the R-value; indicates how close (distance metric) the word segmentation performance is to an ideal point of operation (100% HR with 0% OS).

    Parameters
    ----------
    precision : float
        How often word segmentation correctly predicts a word boundary.
    recall : float
        How often word segmentation's prediction matches a ground truth word boundary.

    Return
    ------
    output : float
        R-Value
    """

    os = get_os(precision, recall)
    r1 = np.sqrt((1 - recall)**2 + os**2)
    r2 = (-os + recall - 1)/np.sqrt(2)

    rvalue = 1 - (np.abs(r1) + np.abs(r2))/2
    return rvalue