# 관련 library 싹다 import 해오기
import os
import sys
import copy
import time
import itertools
from glob import glob
from collections import namedtuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from graphviz import Digraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import torchsummary
from torch.optim import optimizer

# 미리 설정
CONFIG = {
    'unrolled' : False,
    'auxiliary' : True,
    'auxiliary_weight':0.4,
    'drop_path_prob':0.2,
    'init_C':16,
    'n_layers':8,
    'lr':0.025,
    'momentum':0.9,
    'weight_decay':3e-4,
    'grad_clip':5,
    'batch_size':64,
    'search_epoch_size':10,
    'epoch_size':200,
}

# 데이터 로드해서 반환하는 함수
def create_loader(n_valid):
  train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914,0.4822,0.4465),(0.2479,0.2439,0.2616)),])
  
  test_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.4914,0.4822,0.4465),(0.2479,0.2439,0.2616)),])
    
  train_dataset = datasets.CIFAR10('./data', train=True, download = True, transform=train_transform)
  test_dataset = datasets.CIFAR10('./data', train=True, download = True, transform=test_transform)
  train_dataset, valid_dataset = random_split(train_dataset, [50000-n_valid, n_valid])
  train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle= True)
  valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle= True)
  test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle= False)

  return train_loader, valid_loader, test_loader  

train_loader, valid_loader, test_loader = create_loader(n_valid=5000)
x, y = next(iter(train_loader))  


# 데이터 로드해서 반환하는 함수
def create_loader(n_valid):
  train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914,0.4822,0.4465),(0.2479,0.2439,0.2616)),])
  
  test_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.4914,0.4822,0.4465),(0.2479,0.2439,0.2616)),])
    
  train_dataset = datasets.CIFAR10('./data', train=True, download = True, transform=train_transform)
  test_dataset = datasets.CIFAR10('./data', train=True, download = True, transform=test_transform)
  train_dataset, valid_dataset = random_split(train_dataset, [50000-n_valid, n_valid])
  train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle= True)
  valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle= True)
  test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle= False)

  return train_loader, valid_loader, test_loader  

train_loader, valid_loader, test_loader = create_loader(n_valid=5000)
x, y = next(iter(train_loader))  


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


# 각 cell은 여러 mixed op 를 노드로 하는 DAG 로 구성된다
# reduction 인자를 받아서 해당 cell이 reduction cell인지 normal cell인지 구분
# reduction cell을 적용한 이유는 feature 사이즈의 수축을 유도하여 모델의 전체 사이즈를 줄일 수 있기 때문
class SearchCell(nn.Module):
  def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):   # p는 previous를 의미 
    super().__init__()
    self.n_nodes = n_nodes
    if(reduction_p):  # 이전 cell이 reduction cell이라면 현재 입력데이터 사이즈는 이전 cell의
    # 출력데이터 사이즈와 맞지 않는다. 따라서 아래와 같은 절차를 거쳐야 한다. 
    # 근데 이 연산은 논문에 자세히 나와있지 않아서 github의 코드를 참고하였다.
      self.prep0 = FactorizedReduce(C_pp, C, False)
    else:
      self.prep0 = StdConv(C_p, C, 1,1,0,False)

    self.prep1 = StdConv(C_p, C, 1,1,0, False)

    self.DAG = nn.ModuleList()
    for i in range(self.n_nodes):
      self.DAG.append(nn.ModuleList())
      for j in range(i+2):
        if reduction and j<2: # reduction인자를 받아서 해당 cell이 reduction cell이면
          stride = 2 #  stride = 2
        else:
          stride = 1
        op = MixedOp(C, stride)
        self.DAG[i].append(op)
  
  # forward시에 각 mixedop들의 가중치 alpha값을 입력으로 받아 넣는다.
  # 가중합 연산은 mixedop에서 이루어지며 중간노드들의 결과를 concat한 것이 최종 output이다.
  def forward(self, s0, s1, weights):
    s0 = self.prep(s0)
    s1 = self.prep(s1)

    states = [s0, s1]
    for edges, w_list in zip(self.DAG, weights):
      s_cur = sum(edges[i](s,w) for i, (s,w) in enumerate(zip(states, w_list)))
      states.append(s_cur)

    s_out = torch.cat(states[2:], dim=1)
    return s_out



# 전체의 3분의 1,2지점에 reduction cell을 넣어준다.
class Network(nn.Module):
  def __init__(self, C, n_classes, n_layers, criterion, device, steps=4, multiplier =4, stem_multiplier=3): 
    super().__init__()
    #self.C_in = C_in #인풋채널수
    self.C = C  #첫번째 모델 채널수
    self.n_classes = n_classes  # 클래스 개수
    self.n_layers = n_layers  # 레이어 개수
    self.n_nodes = steps    # 셀 내부의 중간노드 개수 
    self.criterion = criterion

    C_cur = stem_multiplier*C
    self.stem = nn.Sequential(
        nn.Conv2d(3, C_cur, 3,1,1,bias = False),
        nn.BatchNorm2d(C_cur)
    )

    # C_pp, C_p는 output channel size, C_cur는 input channel size
    C_pp, C_p, C_cur = C_cur, C_cur, C
    self.cells = nn.ModuleList()
    reduction_p = False
    for i in range(n_layers):
      if i in [n_layers//3, 2*n_layers//3]: # 1/3, 2/3지점에 reduce featuremap size, double channels
        C_cur *= 2
        reduction = True
      else:
        reduction = False
      cell = SearchCell(steps, C_pp, C_p, C_cur, reduction_p, reduction)
      reduction_p = reduction
      self.cells.append(cell)
      C_cur_out = C_cur*steps
      C_pp, C_p = C_p, C_cur_out

    self.gap = nn.AdaptiveAvgPool2d(1)
    self.linear = nn.Linear(C_p, n_classes)

  #모든 cell이 동일한 alpha 값을 가지기때문에 network class에서 관리, network는 
  #alpha를 parameter로 갖는다.
  #아래 함수는 alpha를 관리하는 함수, alpha != weights이다.
  def _init_alphas(self):
        n_ops = len(PRIMITIVES)
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(self.n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

  # forward시에 softmax를 통해서 각 연산의 가중치를 계산하여 넣어준다
  def forward(self, x):
        s0 = s1 = self.stem(x)

        #softmax를 통해서 각 연산의 가중치를 계산하여 넣어준다
        for i, cell in enumerate(self.cells):
          if cell.reduction:
            weights = F.softmax(self.alpha_reduce, dim = -1)
          else:
            weights = F.softmax(self.alpha_normal, dim=-1)
          s0, s1 = s1, cell(s0, s1, weights)

        logits = self.classifier(s1)
        return logits

    
  def weights(self):
        for a, b in self.named_parameters():
            if 'alpha' not in a:
                yield b

  def alphas(self):
        for a, b in self.named_parameters():
            if 'alpha' in a:
                yield b
                
  def loss(self, X, y):
        logits = self(X)
        return self.criterion(logits, y)

  # concat all intermediate nodes 코드를 구현해야함
  # def genotype(self):
 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss().to(device)
model = Network(CONFIG['init_C'], 10, CONFIG['n_layers'], criterion, device).to(device)
print(model)
