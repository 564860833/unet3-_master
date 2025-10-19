# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import unetConv2
from init_weights import init_weights
from deform_conv_v2 import DeformConv2d
from resnet import ResidualBlock


'''
   残差连接
'''
class PreActivateDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.double_conv(x)

class PreActivateResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResUpBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x) + self.ch_avg(x)


class PreActivateResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.ch_avg(x)
        out = self.double_conv(x)
        out = out + identity
        return out

'''
    边缘增强模块
'''
class Edgeh(nn.Module):
   def __init__(self, inchannels1, inchannels2, outchannels1, outchannels2):
       super(Edgeh, self).__init__()
       # 特征图1
       self.conv1 = nn.Conv2d(inchannels1, outchannels1, kernel_size=1, padding=1)
       self.conv2 = nn.Conv2d(outchannels1, inchannels1, kernel_size=3, padding=1)

       # 特征图2
       # 经过上采样把特征图2输出和特征图1一样的尺寸
       self.up = nn.Upsample(scale_factor=2, mode='bilinear')
       self.up_conv = nn.Conv2d(inchannels2, inchannels1, 3, padding=1)
       self.up_bn = nn.BatchNorm2d(inchannels1)
       self.up_relu = nn.ReLU(inplace=True)

       self.conv3 = nn.Conv2d(inchannels1, outchannels1, kernel_size=1, padding=1)
       self.conv4 = nn.Conv2d(outchannels1, inchannels1, kernel_size=3, padding=1)
       self.conv_1 = nn.Conv2d(outchannels2, inchannels1, 3, padding=1)  # 16
       self.bn_1 = nn.BatchNorm2d(inchannels1)
       self.relu_1 = nn.ReLU(inplace=True)

   def forward(self, x1, x2):
        h1 = self.conv1(x1)
        h1 = self.conv2(h1)
        h2 = self.up_relu(self.up_bn(self.up_conv(self.up(x2))))

        h2 = self.conv3(h2)
        h2 = self.conv4(h2)
        out = self.relu_1(self.bn_1(self.conv_1(
            torch.cat((h1, h2), 1))))
        return out

'''
    ASPP模块以及特征融合模块
'''
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        rates = [6, 12, 18]
        # 定义ASPP模块中的空洞卷积操作
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[0], padding=rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[1], padding=rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rates[2], padding=rates[2])
        self.conv3x3_4 = ASPPPooling(in_channels, out_channels)

        # 定义ASPP模块的输出处理
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        # 对输入特征图进行五个不同空洞率的卷积操作
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        feat5 = self.conv3x3_4(x)
        # H, W = feat4.size(2), feat4.size(3)
        #
        # # 将特征图5的尺寸转为和其余特征图一样的尺寸
        # feat5 = F.interpolate(feat5, size=(H, W), mode='bilinear')
        #
        # # softmax函数获得特征掩码
        # weights1 = torch.softmax(feat1, dim=1)
        # weights2 = torch.softmax(feat2, dim=1)
        # weights3 = torch.softmax(feat3, dim=1)
        # weights4 = torch.softmax(feat4, dim=1)
        # weights5 = torch.softmax(feat5, dim=1)
        # # 使得特征掩码和为1
        # weights1 = weights1 / weights1.sum(dim=1, keepdim=True)
        # weights2 = weights2 / weights2.sum(dim=1, keepdim=True)
        # weights3 = weights3 / weights3.sum(dim=1, keepdim=True)
        # weights4 = weights4 / weights4.sum(dim=1, keepdim=True)
        # weights5 = weights5 / weights5.sum(dim=1, keepdim=True)
        # # 为每个特征图分配权重
        # feat1 = feat1 * weights1
        # feat2 = feat2 * weights2
        # feat3 = feat3 * weights3
        # feat4 = feat4 * weights4
        # feat5 = feat5 * weights5

        output = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        # 对拼接后的特征图进行处理（1*1卷积）
        output = self.conv_out(output)

        return output


'''
    UNet 3+
'''
class UNet_3Plus(nn.Module):
    def __init__(self, args):
        super(UNet_3Plus, self).__init__()
        self.args = args
        in_channels = 4
        n_classes = 3
        feature_scale = 4
        is_deconv = True
        is_batchnorm = True
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 192, 512, 1024]

        self.DeformConv2d =DeformConv2d(self.in_channels, filters[0])
        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.aspp = ASPP(filters[4], filters[4])

        # self.Edge = Edgeh(filters[0], filters[1], filters[0], filters[1])


        # self.cam = channel(filters[1])
        ## -------------Decoder--------------
        self.CatChannels = filters[0]  # 连接通道数量PreActivateResBlock
        self.CatBlocks = 5  # 连接块数量
        self.UpChannels = self.CatChannels * self.CatBlocks  # 上采样后的通道数量

        '''stage 4d'''
        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(192, 192, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(192)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(192, 192, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(192)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(192, 192, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(192)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(128, 128, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(128)
        self.relu1d_1 = nn.ReLU(inplace=True)


        self.conv1d_0 = nn.Conv2d(192, 128, 3, padding=1)  # 16
        self.bn1d_0 = nn.BatchNorm2d(128)
        self.relu1d_0 = nn.ReLU(inplace=True)
        # output
        self.outconv1 = nn.Conv2d(128, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.DeformConv2d(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256


        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        hd5 = self.aspp(hd5)
        # out1 = self.Edge(h1, h2)
        ## -------------Decoder-------------
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        # h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1), 1))))  # hd1->320*320*UpChannels
        # out1 = torch.nn.functional.interpolate(out1, size=(160, 160), mode='bilinear', align_corners=False)  # 将特征图的尺寸插值到 [1, 16, 160, 160]
        # hd1 = self.relu1d_0(self.bn1d_0(self.conv1d_0(torch.cat((hd1, out1), 1))))
        d1 = self.outconv1(hd1)  # d1->320*320*n_classes

        return d1
