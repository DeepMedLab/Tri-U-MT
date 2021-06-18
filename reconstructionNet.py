import math
from typing import Iterable, Any

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch import Tensor
from layers import *
from networks import weights_init
import itertools


class SDFNet(nn.Module):
    def __init__(self, nin, nout, l_rate, nG=64, has_dropout=False):
        super().__init__()
        self.encoder = SharedEncoder(nin, nout, has_dropout=has_dropout).cuda()
        self.seg_decoder = SegmentationDecoder(nin, nout).cuda()
        self.sdf_decoder = ReconstructionDecoder(nin, nout).cuda()
        self.rec_decoder = ReconstructionDecoderWoSkip(nin, nout).cuda()
        self.conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        # self.mlp = MLP(32*32*512, 1024)

        self.encoder.apply(weights_init)
        self.seg_decoder.apply(weights_init)
        self.sdf_decoder.apply(weights_init)
        self.rec_decoder.apply(weights_init)

        # optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.conv.parameters()), lr=l_rate, betas=(0.9, 0.99), amsgrad=False)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=l_rate, betas=(0.9, 0.99), amsgrad=False)
        optimizer1 = torch.optim.Adam(self.seg_decoder.parameters(), lr=l_rate, betas=(0.9, 0.99), amsgrad=False)
        optimizer2 = torch.optim.Adam(self.sdf_decoder.parameters(), lr=l_rate/10, betas=(0.9, 0.99), amsgrad=False)
        optimizer3 = torch.optim.Adam(self.rec_decoder.parameters(), lr=l_rate/10, betas=(0.9, 0.99), amsgrad=False)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.98)
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.98)
        scheduler3 = torch.optim.lr_scheduler.ExponentialLR(optimizer3, gamma=0.98)

        # optimizer4 = torch.optim.Adam(self.mlp.parameters(), l_rate=l_rate, betas=(0.9, 0.99), amsgrad=False)
        self.optimizers = [optimizer, optimizer1, optimizer2, optimizer3]
        self.schedulers = [scheduler, scheduler1, scheduler2, scheduler3]

    def forward(self, input):
        feature, x0, x1, x2 = self.encoder(input)
        pred_logits = self.seg_decoder(feature, x0, x1, x2)
        sdf_probs = self.sdf_decoder(feature, x0, x1, x2)
        rec_probs = self.rec_decoder(feature)
        # return torch.flatten(self.conv(feature), start_dim=1), pred_logits, sdf_probs, rec_probs
        return torch.flatten(feature, start_dim=1), pred_logits, sdf_probs, rec_probs


    def optimize(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def scheduler_step(self):
        # for scheduler in self.schedulers:
        #     scheduler.step()
        pass


class MLP(nn.Module):
    def __init__(self, nin, nout, nG=64, has_dropout=False):
        super().__init__()
        self.fs1 = torch.nn.Linear(nin, nout)
        self.fs2 = torch.nn.Linear(nout, nout)

    def forward(self, input):
        input = input.float()
        x1 = self.fs1(input)
        x2 = self.fs2(x1)
        return x2


class SharedEncoder(nn.Module):
    def __init__(self, nin, nout, nG=64, has_dropout=False):
        super().__init__()
        self.has_dropout = has_dropout
        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))

        self.bridge = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                    residualConv(nG * 8, nG * 8),
                                    convBatch(nG * 8, nG * 8))
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, input):
        input = input.float()
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        result = self.bridge(x2)
        if self.has_dropout:
            result = self.dropout(result)
        return result, x0, x1, x2


class SegmentationDecoder(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.unetfinal = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input, feature_scale0, feature_scale1, feature_scale2):
        task1_y0 = self.deconv1(input)
        task1_y1 = self.deconv2(self.conv5(torch.cat((task1_y0, feature_scale2), dim=1)))
        task1_y2 = self.deconv3(self.conv6(torch.cat((task1_y1, feature_scale1), dim=1)))
        task1_y3 = self.conv7(torch.cat((task1_y2, feature_scale0), dim=1))
        task1_result = self.unetfinal(task1_y3)
        return task1_result


class ReconstructionDecoder(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.unetfinal = nn.Conv2d(nG, 2, kernel_size=1)

    def forward(self, input, feature_scale0, feature_scale1, feature_scale2):
        task1_y0 = self.deconv1(input)
        task1_y1 = self.deconv2(self.conv5(torch.cat((task1_y0, feature_scale2), dim=1)))
        task1_y2 = self.deconv3(self.conv6(torch.cat((task1_y1, feature_scale1), dim=1)))
        task1_y3 = self.conv7(torch.cat((task1_y2, feature_scale0), dim=1))
        task1_result = self.unetfinal(task1_y3)
        return F.sigmoid(task1_result)


class ReconstructionDecoderWoSkip(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 8, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 4, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 2, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.unetfinal = nn.Conv2d(nG, 2, kernel_size=1)

    def forward(self, input):
        task1_y0 = self.deconv1(input)
        task1_y1 = self.deconv2(self.conv5(task1_y0))
        task1_y2 = self.deconv3(self.conv6(task1_y1))
        task1_y3 = self.conv7(task1_y2)
        task1_result = self.unetfinal(task1_y3)
        return F.sigmoid(task1_result)