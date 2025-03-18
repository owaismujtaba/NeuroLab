import os
from pathlib import Path

CUR_DIR = os.getcwd()
DATA_DIR = Path(CUR_DIR, 'Data')
MODEL_DIR = Path(CUR_DIR, 'Results', 'TrainedModels')


EEG_SR = 1024
AUDIO_SR = 48000
REF = 'ESR'
BAND = 'HFA'

WIN_LENGHT = 0.05
FRAME_SHIFT = 0.01
MODEL_ORDER = 10
STEP_SIZE = 5
N_FILTERS = 23


EPOCHS = 500
BATCH_SIZE = 512
