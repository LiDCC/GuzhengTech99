import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import numpy as np
from function.lib import *
from function.config import *
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
import math
from visdom import Visdom


class Trainer:
    def __init__(self, model, lr, epoch, save_fn, validation_interval, save_interval):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        self.validation_interval = validation_interval
        self.save_interval = save_interval

    def Tester(self, loader, b_size, we):

        all_pred = np.zeros((b_size, NUM_LABELS, int(LENGTH)))
        all_tar = np.zeros((b_size, NUM_LABELS, int(LENGTH)))
        loss_IPT = 0.0

        self.model.eval()
        ds = 0
        for _, _input in enumerate(loader):
            data, target = Variable(_input[0].to(
                device)), Variable(_input[1].to(device))
            frame_pred = self.model(data)

            loss = sp_loss(frame_pred, target, we)
            loss_IPT += loss.data

            all_tar[ds: ds + len(target)] = target.data.cpu().numpy()
            all_pred[ds: ds + len(target)] = F.sigmoid(
                torch.squeeze(frame_pred)).data.cpu().numpy()
            ds += len(target)

        threshold = 0.5
        pred_inst = np.transpose(all_pred, (1, 0, 2)).reshape(
            (NUM_LABELS, -1))  # shape = [10, 8424] , 8424 = 27*312
        tar_inst = np.transpose(all_tar, (1, 0, 2)).reshape(
            (NUM_LABELS, -1))  # shape = [10, 8424] , 8424 = 27*312
        pred_inst[pred_inst > threshold] = 1
        pred_inst[pred_inst <= threshold] = 0

        metrics = compute_metrics(pred_inst, tar_inst)
        return loss_IPT / b_size, metrics['metric/IPT_frame/precision'][0], metrics['metric/IPT_frame/recall'][0], metrics['metric/IPT_frame/f1'][0]

    def fit(self, tr_loader, va_loader, we):
        st = time.time()

        save_dict = {}
        save_dict['tr_loss'] = []

        lr = self.lr
        optimizer = optim.SGD(self.model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=1e-4)

        lrf = 0.01
        epochs = 100

        def lf(x): return ((1 + math.cos(x * math.pi / epochs)) / 2) * \
            (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        viz = Visdom()
        viz.line([[0., 0.]], [0], win='IPT_loss', opts=dict(
            title='IPT_loss', legend=['train_loss', 'valid_loss']))
        viz.line([[0.]], [0], win='IPT_precision', opts=dict(
            title='IPT_precision', legend=['valid_IPT_precision']))
        viz.line([[0.]], [0], win='IPT_recall', opts=dict(
            title='IPT_recall', legend=['valid_IPT_recall']))
        viz.line([[0.]], [0], win='IPT_F1', opts=dict(
            title='IPT_F1', legend=['valid_IPT_F1']))
        best_acc = 0

        for e in range(1, self.epoch+1):

            loss_total = 0
            print('\n==> Training Epoch #%d lr=%4f' % (e, lr))
            # Training
            for batch_idx, _input in enumerate(tr_loader):
                self.model.train()
                data, target = Variable(_input[0].to(
                    device)), Variable(_input[1].to(device))

                # start feed in
                frame_pred = self.model(data)

                # counting loss
                loss = sp_loss(frame_pred, target, we)
                loss_total += loss.data
                optimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(self.model.parameters(), 3)

                optimizer.step()
                scheduler.step()

                # fresh the board
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d]\tLoss %4f\tTime %d'
                                 % (e, self.epoch, batch_idx+1, len(tr_loader),
                                    loss.data, time.time() - st))
                sys.stdout.flush()

            print('\n')
            print(loss_total/len(tr_loader))

            if e % self.validation_interval == 1:
                print(self.save_fn)
                eva_result = self.Tester(va_loader, len(va_loader.dataset), we)
                self.model.train()

                viz.line([[float(loss_total/len(tr_loader.dataset)),
                         float(eva_result[0])]], [e - 1], win='IPT_loss', update='append')
                viz.line([[float(eva_result[1])]], [e - 1],
                         win='IPT_precision', update='append')
                viz.line([[float(eva_result[2])]], [e - 1],
                         win='IPT_recall', update='append')
                viz.line([[float(eva_result[3])]], [e - 1],
                         win='IPT_F1', update='append')
                if eva_result[3] > best_acc:
                    best_acc = eva_result[3]
                    save_dict['state_dict'] = self.model.state_dict()
                    torch.save(save_dict, self.save_fn + 'best')

            if e % self.save_interval == 1:
                save_dict['state_dict'] = self.model.state_dict()
                torch.save(save_dict, self.save_fn+'_e_%d' % (e-1))
