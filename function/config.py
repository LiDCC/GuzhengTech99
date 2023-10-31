# config
import torch

status = 'inst'
data_name = 'gt'
model_name = 'SY'

saveName = 'baseline'  # name of the model to save and load
FRE = 88  # number of frequency bin
TIME_LENGTH = 3  # 3 seconds
LENGTH = 258  # number of frame in 3 seconds,258
NUM_LABELS = 7  # number of instrument
BATCH_SIZE = 10
SAMPLE_RATE = 44100
MIN_MIDI = 21
MAX_MIDI = 108
HOP_LENGTH = 512  # SAMPLE_RATE * ZHEN_LENGTH // 1000

# for evaluation
model_choose = 55  # number of model to load
# whether to calculate score matrix (F1-score, Precision and Recall)
isPre = True
isDraw = True  # whether to draw the result
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
