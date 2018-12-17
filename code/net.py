# -*- coding:utf-8 -*-
from mxnet.gluon import nn
from mxnet import init
import mxnet as mx

class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, x, *args, **kwargs):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(Y + x)

def resnet34(num_classes):
    net=nn.HybridSequential()
    net.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3))
    net.add(nn.BatchNorm())
    net.add(nn.Activation('relu'))
    net.add(nn.MaxPool2D(pool_size=3,strides=2,padding=1))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64,3,first_block=True))
    net.add(resnet_block(128,4))
    net.add(resnet_block(256,6))
    net.add(resnet_block(512,3))
    net.add(nn.GlobalAvgPool2D())
    net.add(nn.Dense(num_classes))
    return net

def get_net(ctx,num_classes):
    net=resnet34(num_classes)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net