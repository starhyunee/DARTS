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
from operation import FactorizedReduce, StdConv, MixedOp

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
