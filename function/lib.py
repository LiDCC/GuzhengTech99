import sys
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from function.config import *
from collections import defaultdict
from scipy.stats import hmean
from mir_eval.util import midi_to_hz
from mir_eval.multipitch import evaluate as evaluate_frames

# Dataset


class Data2Torch(Dataset):
    def __init__(self, data):

        self.X = data[0]
        self.Y = data[1]

    def __getitem__(self, index):

        mX = torch.from_numpy(self.X[index]).float()
        mY = torch.from_numpy(self.Y[index]).float()
        return mX, mY

    def __len__(self):
        return len(self.X)


def sp_loss(fla_pred, target, gwe):

    we = gwe.to(device)
    wwe = 1
    we *= wwe

    loss = 0

    for idx, (out, fl_target) in enumerate(zip(fla_pred, target)):
        twe = we.view(-1, 1).repeat(1, fl_target.size(1)
                                    ).type(torch.cuda.FloatTensor)
        ttwe = twe * fl_target.data + (1 - fl_target.data) * wwe
        loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, reduction='mean')
        # print(target.shape)
        loss += loss_fn(torch.squeeze(out), fl_target)

    return loss


def num_params(model):
    params = 0
    for i in model.parameters():
        params += i.view(-1).size()[0]
    print('#params:%d' % (params))


def notes_to_frames(roll):

    time = np.arange(roll.shape[-1])
    freqs = [roll[:, t].nonzero()[0] for t in time]
    return time, freqs


def compute_metrics(pred_inst, Yte):
    metrics = defaultdict(list)
    eps = sys.float_info.epsilon
    t_ref, f_ref = notes_to_frames(Yte)
    t_est, f_est = notes_to_frames(pred_inst)

    scaling = HOP_LENGTH / SAMPLE_RATE

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [np.array([midi_to_hz(MIN_MIDI + midi)
                      for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi)
                      for midi in freqs]) for freqs in f_est]

    IPT_frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/IPT_frame/f1'].append(
        hmean([IPT_frame_metrics['Precision'] + eps, IPT_frame_metrics['Recall'] + eps]) - eps)

    for key, loss in IPT_frame_metrics.items():
        metrics['metric/IPT_frame/' + key.lower().replace(' ', '_')
                ].append(loss)

    return metrics
