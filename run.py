import torch.optim as optim
import datetime
date = datetime.datetime.now()
import sys
sys.path.append('./function')
from function.lib import *
from function.fit import *
from function.model import *
from function.load_data import *
from function.config import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_weight(Ytr):#(2493, 258, 6)
	mp = Ytr[:].sum(0).sum(0) #(6,)
	mmp = mp.astype(np.float32) / mp.sum()
	cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
	inverse_feq = torch.from_numpy(cc)
	return inverse_feq

out_model_fn = './data/model/%d%d%d%d:%d:%d/%s/'%(date.year,date.month,date.day,date.hour,date.minute,date.second,saveName)
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

# load data
wav_dir = 'Guzheng_TechPitch/data'
csv_dir = 'Guzheng_TechPitch/labels'
groups = ['train']
vali_groups = ['validation']
Xtr,Ytr,avg,std = load(wav_dir,csv_dir,groups)
Xva,Yva,va_avg,va_std = load(wav_dir,csv_dir,vali_groups,avg,std)
print ('finishing data loading...')

# Build Dataloader
t_kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 2, 'pin_memory': True, 'drop_last': True}
v_kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 10, 'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(Data2Torch([Xtr[:], Ytr[:]]), shuffle=True, **t_kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch([Xva, Yva]), **v_kwargs)
print ('finishing data building...')

#Construct Model
model = SY_multi_scale_attn222().to(device)
print (model)
num_params(model)
print ('batch_size:%d num_labels:%d'%(BATCH_SIZE, NUM_LABELS))
print ('Dataset:' + data_name)
print ('Xtr:' + str(Xtr.shape))
print ('Xte:' + str(Xva.shape))
print ('Ytr:' + str(Ytr.shape))
print ('Yte:' + str(Yva.shape))
inverse_feq = get_weight(Ytr.transpose(0,2,1))

#Start training
Trer = Trainer(model, 0.01, 10000, out_model_fn, avg,std, validation_interval=5, save_interval=10)
Trer.fit(tr_loader, va_loader,inverse_feq)

