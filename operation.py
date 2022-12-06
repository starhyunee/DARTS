#이 파일은 mixed operation을 비롯한 여러 연산들의 class를 정의한 파일
import itertools
from glob import glob
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import optimizer
# operation class 정의

#dilated convolution
class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

# 항등, skip connect
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# separated convolution
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

# 이건 논문엔 없는 것 같은데 github엔 있어서 참고한 class
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

# None class
class Zero(nn.Module):
  def __init__(self,stride):
      super(Zero, self).__init__()
      self.stride = stride
 
  def forward(self, x):
      if self.stride == 1:
        return x.mul(0.)
      return x[:, :, ::self.stride, ::self.stride].mul(0.)
  
# standard convolution class
class StdConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


# OPS dictionary, 위의 연산들을 바탕으로 작성
OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(kernel_size=3, stride=stride, padding =1, count_include_pad =False),
    'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(kernel_size=3, stride=stride, padding = 1),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, affine),
    'skip_connect' : lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine)
}

# Mixedop를 생성하기 전에 우선 후보 연산들을 정의
PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'] 

# 모든 후보 연산들이 믹스되어 있는 연산 mixedop
# 이 mixedop이 모여서 DAG로 표현되는 하나의 cell이 되고 
# 이 cell이 모여서 network를 이룬다.
class MixedOp(nn.Module):
  def __init__(self, C, stride):
    super().__init__()
    self.ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      self.ops.append(op)

  # forward 시에는 weight를 인자로 받아서 가중합을 계산한다
  def forward(self, x, weights):
    return sum(w*op(x) for w, op in zip(weights, self.ops))