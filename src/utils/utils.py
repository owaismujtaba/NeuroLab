import numpy as np
from scipy.stats import pearsonr
import pdb

def calculate_pcc_spectrgorams(pred, actual):
    """
        Computes the mean Pearson correlation coefficient (PCC) across spectral bins 
        between predicted and actual spectrograms.


        Parameters:
            pred (np.ndarray): Predicted spectrogram of shape (N, 23), where N is the number 
                            of samples and 23 is the number of spectral bins.
            actual (np.ndarray): Actual spectrogram of the same shape as pred.

        Returns:
            float: Mean Pearson correlation coefficient across spectral bins.
    """
    pcc_values = []
    for spectral_bin in range(pred.shape[1]):
        pred_bin = pred[:, spectral_bin]
        actual_bin = actual[:, spectral_bin]
        r, p = pearsonr(pred_bin, actual_bin)
        pcc_values.append(r)
    
    return np.mean(pcc_values)
    


def z_score_normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the input data using Z-score normalization.
    
    Parameters:
        data (np.ndarray): Input data of shape (N, 2247), where N is the number of samples.
        
    Returns:
        np.ndarray: Z-score normalized data with the same shape as input.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    normalized_data = (data - mean) / std
    return normalized_data