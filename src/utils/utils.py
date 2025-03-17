import numpy as np
from scipy.stats import pearsonr
import pdb

def calculate_pcc_spectrgorams(pred, actual):
    pcc_values = []
    for spectral_bin in range(pred.shape[1]):
        pred_bin = pred[:, spectral_bin]
        actual_bin = actual[:, spectral_bin]
        r, p = pearsonr(pred_bin, actual_bin)
        pcc_values.append(p)
    
    return np.mean(pcc_values)
    


def normalize_eeg(data):
    pdb.set_trace()