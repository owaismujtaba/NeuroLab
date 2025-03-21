import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

import config as config

import pdb

def plot_pcc_across_subjects(model_name='NeuralNetwork'):

    data_dir = config.MODEL_DIR

    results = {}
    for subject in range(21, 31):
        sub_dir = Path(data_dir, str(subject))
        model_dir = Path(sub_dir,model_name)
        files = os.listdir(model_dir)

        pcc_values = []
        for file in files:
            file_dir = Path(model_dir, file)
            file_data = np.load(file_dir)
            pcc_values.append(float(file_data[-1]))
        results[subject] = pcc_values


        data =results
    subjects = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=values, zorder=1)

    for i, subject_values in enumerate(values):
        mean_value = np.mean(subject_values)
        plt.plot(i, mean_value, marker='s', color='red', markersize=8, label='Mean' if i == 0 else "")

    for i, subject_values in enumerate(values):
        plt.scatter([i] * len(subject_values), subject_values, color='black', alpha=0.6, zorder=2, label='Data points' if i == 0 else "")

    # Customize the plot
    plt.xticks(ticks=np.arange(len(subjects)), labels=subjects)
    plt.xlabel("Subjects")
    plt.ylabel("PCC")
    plt.savefig(f'{model_name}_pcc.png', dpi=800)
            


def plot_spectrograms(actual, predictions):
    # Plot the actual spectrogram

    '''
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(actual, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
    plt.colorbar(label='Spectral intensity')
    plt.title('Actual Spectrogram')
    plt.xlabel('Spectral Bins')
    plt.ylabel('Time Windows')

    # Plot the predicted spectrogram
    plt.subplot(1, 2, 2)
    plt.imshow(predictions, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
    plt.colorbar(label='Spectral intensity')
    plt.title('Predicted Spectrogram')
    plt.xlabel('Spectral Bins')
    plt.ylabel('Time Windows')

    # Show the plots
    plt.tight_layout()
    plt.savefig('compariosn.png', dpi=600)
    pdb.set_trace()'
    '''''
    np.save('actual.npy', actual)
    np.save('predicted.npy', predictions)
