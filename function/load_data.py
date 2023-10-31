import numpy as np
from glob import glob
import os
import csv
import librosa
from tqdm import tqdm
from function.norm_lib import *
from function.config import *
import torch


def norm(avg, std, data, size):
    avg = np.tile(avg.reshape((1, -1, 1, 1)), (size[0], 1, size[2], size[3]))
    std = np.tile(std.reshape((1, -1, 1, 1)), (size[0], 1, size[2], size[3]))
    data = (data - avg)/std
    return data


def load(wav_dir, csv_dir, groups, avg=None, std=None):
    # Return all [(audio address, corresponding to csv file address), ( , ), ...] list
    if std is None:
        std = np.array([None])
    if avg is None:
        avg = np.array([None])

    def files(wav_dir, csv_dir, group):
        flacs = sorted(glob(os.path.join(wav_dir, group, '*.flac')))
        if len(flacs) == 0:
            flacs = sorted(glob(os.path.join(wav_dir, group, '*.wav')))

        csvs = sorted(glob(os.path.join(csv_dir, group, '*.csv')))
        files = list(zip(flacs, csvs))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')
        result = []
        for audio_path, csv_path in files:
            result.append((audio_path, csv_path))
        return result

    # Returns the CQT of the input audio
    def logCQT(file):
        sr = SAMPLE_RATE
        y, sr = librosa.load(file, sr=sr)
        # 帧长为32ms（1000ms/(16000/512) = 32ms）,D2的频率是73.418
        cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH,
                          fmin=27.5, n_bins=88, bins_per_octave=12)
        return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0

    def chunk_data(f):
        s = int(SAMPLE_RATE*TIME_LENGTH/HOP_LENGTH)
        num = FRE
        xdata = np.transpose(f)
        x = []
        length = int(np.ceil((int(len(xdata) / s) + 1) * s))
        app = np.zeros((length - xdata.shape[0], xdata.shape[1]))
        xdata = np.concatenate((xdata, app), 0)
        for i in range(int(length / s)):
            data = xdata[int(i * s):int(i * s + s)]
            x.append(np.transpose(data[:s, :]))

        return np.array(x)

    def load_all(audio_path, csv_path):

        saved_data_path = audio_path.replace(
            '.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        # Load audio features
        # The shape of cqt (88, 8520), 8520 is the number of frames on the time axis
        cqt = logCQT(audio_path)

        # Load the ground truth label
        hop = HOP_LENGTH
        n_steps = cqt.shape[1]
        n_IPTs = NUM_LABELS

        technique = {'chanyin': 0, 'dianyin': 6, 'shanghua': 2, 'xiahua': 3,
                     'huazhi': 4, 'guazou': 4, 'lianmo': 4, 'liantuo': 4, 'yaozhi': 5, 'boxian': 1}

        IPT_label = np.zeros([n_IPTs, n_steps], dtype=int)

        with open(csv_path, 'r') as f:  # csv file for each audio
            print(csv_path)
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:  # each note
                onset = float(label['onset_time'])
                offset = float(label['offset_time'])
                IPT = int(technique[label['IPT']])
                left = int(round(onset * SAMPLE_RATE / hop))
                frame_right = int(round(offset * SAMPLE_RATE / hop))
                frame_right = min(n_steps, frame_right)
                IPT_label[IPT, left:frame_right] = 1

        data = dict(audiuo_path=audio_path, csv_path=csv_path,
                    cqt=cqt, IPT_label=IPT_label)
        torch.save(data, saved_data_path)
        return data

    data = []
    print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} ")
    for group in groups:
        for input_files in tqdm(files(wav_dir, csv_dir, group), desc='Loading group %s' % group):
            data.append(load_all(*input_files))
    i = 0
    for dic in data:
        x = dic['cqt']
        x = chunk_data(x)
        y_i = dic['IPT_label']
        y_i = chunk_data(y_i)

        if i == 0:
            Xtr = x
            Ytr_i = y_i
            i += 1
        else:
            Xtr = np.concatenate([Xtr, x], axis=0)
            Ytr_i = np.concatenate([Ytr_i, y_i], axis=0)

    # Transform the shape of the input
    Xtr = np.expand_dims(Xtr, axis=3)

    # Calculate the mean and variance of the input
    if avg.all() == None and std.all() == None:
        avg, std = RoW_norm(Xtr, './data/%s%d_avg_std' % ('inst', 5))
        print("avg.shape:", avg.shape)
        print("std.shape", std.shape)

    Xtr = norm(avg, std, Xtr, Xtr.shape)  # Normalize

    print("Xtr.shape", Xtr.shape)
    print("Ytr.shape", Ytr_i.shape)
    return Xtr, Ytr_i, avg, std
