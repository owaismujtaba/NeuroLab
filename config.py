import os
from pathlib import Path

CUR_DIR = os.getcwd()
DATA_DIR = Path(CUR_DIR, 'Data')


EEG_SR = 1024
REF = 'ESR'
BAND = 'HFA'

WIN_LENGHT = 0.05
FRAME_SHIFT = 0.01
MODEL_ORDER = 10
STEP_SIZE = 5

