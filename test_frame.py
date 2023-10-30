# from sklearn.metrics import f1_score, precision_score, recall_score
from mlcm import mlcm
import sklearn.metrics as skm
import numpy as np
# from sklearn.metrics import precision_score
from function.load_data import *
from function.lib import *
from function.model import *
import datetime
import matplotlib.pyplot as plt
import sys
# import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
date = datetime.datetime.now()

sys.path.append('./function')
# from audioset import *


def start_test():

    # load model
    model = SY_multi_scale_attn222().to(device)
    save_dic = torch.load(
        'data/model/20229269-30-30_final_model/baseline/_e_860', map_location='cpu')  # 860
    model.load_state_dict(save_dic['state_dict'])
    print('finishing loading model')

    Xavg, Xstd = save_dic['avg'], save_dic['std']
    wav_dir = 'Guzheng_TechPitch/data'
    csv_dir = 'Guzheng_TechPitch/labels'
    test_group = ['test']
    Xte, Yte, _, _ = load(wav_dir, csv_dir, test_group,
                          Xavg.data.cpu().numpy(), Xstd.data.cpu().numpy())
    print('finishing loading dataset')

    # predict configure
    v_kwargs = {'batch_size': 8, 'num_workers': 10, 'pin_memory': True}
    loader = torch.utils.data.DataLoader(Data2Torch([Xte, Yte]), **v_kwargs)
    all_pred = np.zeros((Yte.shape[0], NUM_LABELS, Yte.shape[2]))
    all_tar = np.zeros((Yte.shape[0], NUM_LABELS, Yte.shape[2]))

    # start predict
    print('start predicting...')
    model.eval()
    ds = 0
    for idx, _input in enumerate(loader):
        data, target = Variable(_input[0].to(
            device)), Variable(_input[1].to(device))
        f_pred = model(data, Xavg, Xstd)
        all_tar[ds: ds + len(target)] = target.data.cpu().numpy()
        all_pred[ds: ds + len(target)] = F.sigmoid(torch.squeeze(f_pred)
                                                   ).data.cpu().numpy()  # 最终开主食
        ds += len(target)

    threshold = 0.5
    pred_IPT = np.transpose(all_pred, (1, 0, 2)).reshape((NUM_LABELS, -1))
    pred_IPT[pred_IPT > threshold] = 1
    pred_IPT[pred_IPT <= threshold] = 0
    target_IPT = np.transpose(all_tar, (1, 0, 2)).reshape((NUM_LABELS, -1))

    # compute metrics
    # metrics = compute_metrics(pred_IPT, target_IPT)
    #
    # print("IPT_frame_precision:", metrics['metric/IPT_frame/precision'])
    # print("IPT_frame_recall:", metrics['metric/IPT_frame/recall'])
    # print("IPT_frame_f1:", metrics['metric/IPT_frame/f1'])
    # print("IPT_frame_accuracy:", metrics['metric/IPT_frame/accuracy'])
    #
    # metrics0 = compute_metrics(pred_IPT[0,:].reshape((1,-1)), target_IPT[0,:].reshape((1,-1)))
    # metrics1 = compute_metrics(pred_IPT[1,:].reshape((1,-1)), target_IPT[1,:].reshape((1,-1)))
    # metrics2 = compute_metrics(pred_IPT[2,:].reshape((1,-1)), target_IPT[2,:].reshape((1,-1)))
    # metrics3 = compute_metrics(pred_IPT[3, :].reshape((1,-1)), target_IPT[3, :].reshape((1,-1)))
    # metrics4 = compute_metrics(pred_IPT[4, :].reshape((1,-1)), target_IPT[4, :].reshape((1,-1)))
    # metrics5 = compute_metrics(pred_IPT[5, :].reshape((1,-1)), target_IPT[5, :].reshape((1,-1)))
    # metrics6 = compute_metrics(pred_IPT[6, :].reshape((1,-1)), target_IPT[6, :].reshape((1,-1)))
    #
    # print("vibrato_frame_precision:", metrics0['metric/IPT_frame/precision'])
    # print("vibrato_frame_recall:", metrics0['metric/IPT_frame/recall'])
    # print("vibrato_frame_f1:", metrics0['metric/IPT_frame/f1'])
    # print("vibrato_frame_accuracy:", metrics0['metric/IPT_frame/accuracy'])
    #
    # print("plucks_frame_precision:", metrics1['metric/IPT_frame/precision'])
    # print("plucks_frame_recall:", metrics1['metric/IPT_frame/recall'])
    # print("plucks_frame_f1:", metrics1['metric/IPT_frame/f1'])
    # print("plucks_frame_accuracy:", metrics1['metric/IPT_frame/accuracy'])
    #
    # print("UP_frame_precision:", metrics2['metric/IPT_frame/precision'])
    # print("UP_frame_recall:", metrics2['metric/IPT_frame/recall'])
    # print("UP_frame_f1:", metrics2['metric/IPT_frame/f1'])
    # print("UP_frame_accuracy:", metrics2['metric/IPT_frame/accuracy'])
    #
    # print("DP_frame_precision:", metrics3['metric/IPT_frame/precision'])
    # print("DP_frame_recall:", metrics3['metric/IPT_frame/recall'])
    # print("DP_frame_f1:", metrics3['metric/IPT_frame/f1'])
    # print("DP_frame_accuracy:", metrics3['metric/IPT_frame/accuracy'])
    #
    # print("glissando_frame_precision:", metrics4['metric/IPT_frame/precision'])
    # print("glissando_frame_recall:", metrics4['metric/IPT_frame/recall'])
    # print("glissando_frame_f1:", metrics4['metric/IPT_frame/f1'])
    # print("glissando_frame_accuracy:", metrics4['metric/IPT_frame/accuracy'])
    #
    # print("tremolo_frame_precision:", metrics5['metric/IPT_frame/precision'])
    # print("tremolo_frame_recall:", metrics5['metric/IPT_frame/recall'])
    # print("tremolo_frame_f1:", metrics5['metric/IPT_frame/f1'])
    # print("tremolo_frame_accuracy:", metrics5['metric/IPT_frame/accuracy'])
    #
    # print("PN_frame_precision:", metrics6['metric/IPT_frame/precision'])
    # print("PN_frame_recall:", metrics6['metric/IPT_frame/recall'])
    # print("PN_frame_f1:", metrics6['metric/IPT_frame/f1'])
    # print("PN_frame_accuracy:", metrics6['metric/IPT_frame/accuracy'])

    # save
    name = 'firsttryXXXX'
    np.save('output_data/IPT/' + name[:-4] + '.npy', pred_IPT)
    np.save('output_data/Yte/' + name[:-4] + '.npy', Yte)
    np.save('output_data/all_target/' + name[:-4] + '.npy', target_IPT)
    np.save('output_data/Xte/' + name[:-4] + '.npy', Xte)

    # confusion matrix
    label_true = target_IPT.transpose(-1, -2)
    label_pred = pred_IPT.transpose(-1, -2)
    conf_mat, normal_conf_mat = mlcm.cm(label_true, label_pred)

    # compute float normal_conf_mat
    divide = conf_mat.sum(axis=1, dtype='int64')
    for indx in range(len(divide)):
        if divide[indx] == 0:  # To avoid division by zero
            divide[indx] = 1

    normal_conf_mat2 = np.zeros((len(divide), len(divide)), dtype=np.float64)
    for i in range(len(divide)):
        for j in range(len(divide)):
            normal_conf_mat2[i][j] = (float(conf_mat[i][j]) / divide[i])
    print('\nRaw confusion Matrix:')
    print(conf_mat)
    print('\nNormalized confusion Matrix (%):')
    print(normal_conf_mat)

    mcm = skm.multilabel_confusion_matrix(label_true, label_pred)
    print(mcm)
    print(skm.classification_report(label_true, label_pred, digits=4))

    # draw confusion matrix
    labels_name = ["vibrato", "plucks", "UP",
                   "DP", "glissando", "tremolo", "PN", "NTL"]
    pred_name = ["vibrato", "plucks", "UP", "DP",
                 "glissando", "tremolo", "PN", "NPL"]
    plt.figure(figsize=(8, 5))
    # Only draw the color grid without values
    plt.imshow(normal_conf_mat2, cmap=plt.cm.Blues, aspect=0.3)
    # plt.title("Normalized confusion matrix")  # title
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt.yticks(range(8), labels_name)  # y-axis label
    plt.xticks(range(8), pred_name)  # x-axis label

    for x in range(8):
        for y in range(8):
            # numerical processing
            value = float(format('%.2f' % normal_conf_mat2[y, x]))
            if x == y:
                if x == 6 or x == 3:
                    plt.text(x, y, value, verticalalignment='center',
                             horizontalalignment='center')  # write value
                else:
                    plt.text(x, y, value, verticalalignment='center',
                             horizontalalignment='center', color='white')  # write value
            else:
                plt.text(x, y, value, verticalalignment='center',
                         horizontalalignment='center')  # write value

    # Automatically adjust subplot parameters so that it fills the entire image area
    plt.tight_layout()

    plt.colorbar()  # color bar
    # bbox_inches='tight', it can ensure that the label information is fully displayed
    plt.savefig('./ConfusionMatrix.png', bbox_inches='tight')
    plt.show()


start_test()
