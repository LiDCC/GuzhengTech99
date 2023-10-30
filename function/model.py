import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
sys.path.append('../fun')
import math
from function.config import *

class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp)       
        self.conv1 = nn.Conv2d(inp, out, (3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(out)       
        self.conv2 = nn.Conv2d(out, out, (3,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(out)       

        self.sk = nn.Conv2d(inp, out, (1,1), padding=(0,0))

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        out += self.sk(x)
        return out

class self_att(nn.Module):
    def __init__(self, embeded_dim, num_heads, out):
        super(self_att, self).__init__()
        self.att = nn.MultiheadAttention(embeded_dim, num_heads, batch_first=True)

    def forward(self, x):
        x1 = x.squeeze().transpose(-1,-2) #[batch, T/9, FRE*3]
        res_branch, attn_wei = self.att(x1, x1, x1)
        res = res_branch.transpose(-1,-2).unsqueeze(-1)
        res = torch.add(res, x)
        return res

class SY_multi_scale_attn222(nn.Module):
    def __init__(self):
        super(SY_multi_scale_attn222, self).__init__()
        inp = FRE
        size = 1
        fs = (3, 1)  # kernel_size
        ps = (1, 0)  # padding_size

        self.bn0 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, FRE, (1, size), padding=(0, 0))

        self.maxpool012 = nn.MaxPool2d((3,1), (3,1))
        self.conv02 = nn.Conv2d(inp, FRE, (1,size), padding=(0, 0))

        self.block11 = block(FRE, FRE * 2)
        self.block12 = block(FRE, FRE*2)

        self.maxpool112 = nn.MaxPool2d((3, 1), (3, 1))
        self.dropout12 = nn.Dropout(p=0.2)
        self.maxpool123 = nn.MaxPool2d((3, 1), (3, 1))
        self.dropout123 = nn.Dropout(p=0.2)
        self.us121 = nn.ConvTranspose2d(FRE*2, FRE*2, kernel_size=(3, 1), stride=(3, 1))


        self.conv21 = nn.Conv2d(FRE*2, FRE*2, (1, 2))
        self.conv22 = nn.Conv2d(FRE*2, FRE*2, (1, 2))
        self.conv23 = nn.Conv2d(FRE*2, FRE*2, (1, 1))

        self.block21 = block(FRE * 2, FRE * 3)
        self.block22 = block(FRE * 2, FRE * 3)
        self.block23 = block(FRE * 2, FRE * 3)

        self.self_att23 = self_att(FRE * 3, 1, 1)
        self.bn23 = nn.BatchNorm2d(FRE * 3)

        self.maxpool212 = nn.MaxPool2d((3, 1), (3, 1))
        self.maxpool223 = nn.MaxPool2d((3, 1), (3, 1))
        self.dropout22 = nn.Dropout(p=0.2)
        self.dropout23 = nn.Dropout(p=0.2)
        self.us221 = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(3, 1), stride=(3, 1))
        self.us232 = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(3, 1), stride=(3, 1))

        self.conv31 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))
        self.conv32 = nn.Conv2d(FRE * 3, FRE * 3, (1, 3))
        self.conv33 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))

        self.block31 = block(FRE * 3, FRE * 3)
        self.block32 = block(FRE * 3, FRE * 3)
        self.block33 = block(FRE * 3, FRE * 3)


        self.bn31 = nn.BatchNorm2d(FRE * 3)
        self.relu31 = nn.ReLU(inplace=True)
        self.bn32 = nn.BatchNorm2d(FRE * 3)
        self.relu32 = nn.ReLU(inplace=True)
        self.bn33 = nn.BatchNorm2d(FRE * 3)
        self.relu33 = nn.ReLU(inplace=True)
        self.maxpool312 = nn.MaxPool2d((3, 1), (3, 1))
        self.dropout312 = nn.Dropout(p=0.2)
        self.us321 = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(3, 1), stride=(3, 1))
        self.us332 = nn.ConvTranspose2d(FRE * 3, FRE * 3, kernel_size=(3, 1), stride=(3, 1))

        self.self_att = self_att(FRE * 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(FRE * 3)

        self.conv41 = nn.Conv2d(FRE * 3, FRE * 3, (1, 2))
        self.conv42 = nn.Conv2d(FRE * 3, FRE * 3, (1, 3))

        self.block41 = block(FRE * 3, FRE * 2)
        self.block42 = block(FRE * 3, FRE * 2)

        self.us421 = nn.ConvTranspose2d(FRE * 2, FRE * 2, kernel_size=(3, 1), stride=(3, 1))

        self.conv51 = nn.Conv2d(FRE * 2, FRE * 2, (1, 2))
        self.block51 = block(FRE * 2, FRE)

        self.bn51 = nn.BatchNorm2d(FRE)
        self.relu51 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(FRE, NUM_LABELS, fs, padding=ps)

    def forward(self, x, Xavg, Xstd):
        x0 = self.bn0(x)

        x02 = self.maxpool012(x0)
        x02 = self.block12(self.conv02(x02))

        x1 = self.conv1(x0)
        x11 = self.block11(x1)

        x11_n = torch.cat([x11, self.us121(x02, output_size= x11.shape)], -1)
        x12 = self.dropout12(self.maxpool112(x11))
        x12_n = torch.cat([x02, x12], -1)
        x13 = self.dropout123(self.maxpool123(x02))

        x21 = self.block21(self.conv21(x11_n))
        x22 = self.block22(self.conv22(x12_n))
        x23 = self.block23(self.conv23(x13))

        x21_n = torch.cat([x21, self.us221(x22, output_size = x21.shape)], -1)
        x22_n = torch.cat([x22, self.dropout22(self.maxpool212(x21)),self.us232(x23, output_size=x22.shape)], -1)
        x23 = self.self_att23(x23)
        x23 = self.bn23(x23)
        x23_n = torch.cat([x23, self.dropout23(self.maxpool223(x22))], -1)

        x31 = self.relu31(self.bn31(self.block31(self.conv31(x21_n))))
        x32 = self.relu32(self.bn32(self.block32(self.conv32(x22_n))))
        x33 = self.relu33(self.bn33(self.block33(self.conv33(x23_n))))

        x33 = self.bn4(self.self_att(x33))

        x31_n = torch.cat([x31, self.us321(x32, output_size=x31.shape)], dim=-1)
        x32_n = torch.cat([x32, self.us332(x33, output_size=x32.shape), self.dropout312(self.maxpool312(x31))], dim=-1)

        x41 = self.block41(self.conv41(x31_n))
        x42 = self.block42(self.conv42(x32_n))

        x51_n = torch.cat([x41, self.us421(x42, output_size=x41.shape)], dim=-1)

        x51 = self.relu51(self.bn51(self.block51(self.conv51(x51_n))))

        pred = self.conv4(x51)
        # print(pred.shape)
        return pred