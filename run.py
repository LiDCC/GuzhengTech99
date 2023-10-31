import os
from function.config import *
from function.model import *
from function.fit import *
from function.lib import *
import sys
from datasets import load_dataset
import datetime
date = datetime.datetime.now()
sys.path.append('./function')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_weight(Ytr):  # (2493, 258, 6)
    mp = Ytr[:].sum(0).sum(0)  # (6,)
    mmp = mp.astype(np.float32) / mp.sum()
    cc = ((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
    inverse_feq = torch.from_numpy(cc)
    return inverse_feq


if __name__ == "__main__":
    out_model_fn = './data/model/%d%d%d%d-%d-%d/%s/' % (
        date.year, date.month, date.day, date.hour, date.minute, date.second, saveName)
    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    # load data
    wav_dir = 'Guzheng_TechPitch/data'
    csv_dir = 'Guzheng_TechPitch/labels'

    trainset = load_dataset(
        'H:/workspace/huggingface/Guzheng_Tech99/Guzheng_Tech99.py', split='train')
    validset = load_dataset(
        'H:/workspace/huggingface/Guzheng_Tech99/Guzheng_Tech99.py', split='validation')

    Xtr, Ytr, Xva, Yva = [], [], [], []

    for item in trainset:
        Xtr.append(item['data'])
        Ytr.append(item['label'])

    for item in validset:
        Xva.append(item['data'])
        Yva.append(item['label'])

    Xtr = np.array(Xtr)
    Ytr = np.array(Ytr)
    Xva = np.array(Xva)
    Yva = np.array(Yva)
    print('finishing data loading...')

    # Build Dataloader
    t_kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 2,
                'pin_memory': True, 'drop_last': True}
    v_kwargs = {'batch_size': BATCH_SIZE,
                'num_workers': 10, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(
        Data2Torch([Xtr[:], Ytr[:]]), shuffle=True, **t_kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch([Xva, Yva]), **v_kwargs)
    print('finishing data building...')

    # Construct Model
    model = SY_multi_scale_attn222().to(device)
    print(model)
    num_params(model)
    print('batch_size:%d num_labels:%d' % (BATCH_SIZE, NUM_LABELS))
    print('Dataset:' + data_name)
    print('Xtr:' + str(Xtr.shape))
    print('Xte:' + str(Xva.shape))
    print('Ytr:' + str(Ytr.shape))
    print('Yte:' + str(Yva.shape))
    inverse_feq = get_weight(Ytr.transpose(0, 2, 1))

    # Start training
    Trer = Trainer(model, 0.01, 10000, out_model_fn,
                   validation_interval=5, save_interval=10)
    Trer.fit(tr_loader, va_loader, inverse_feq)
