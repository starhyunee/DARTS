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
from searchcell import SearchCell
from operation import PRIMITIVES

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
 
