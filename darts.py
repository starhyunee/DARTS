import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity  하나의 layer의 output을 몇 개의 layer를 건너뛰고 다음 layer의 input에 추가
    'sep_conv_3x3',#커널을 두 개의 작은 커널로 나누는것
    'sep_conv_5x5',
    'dil_conv_3x3', #한칸씩 띄어서 필터를 적용?
    'dil_conv_5x5',
    'none'
]

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
}
class PoolBN(nn.Module):
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out

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

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.


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


class MixedOp(nn.Module):
  def __init__(self, C, stride):
    super().__init__()
    self._ops = nn.ModuleList()
    for primitive in gt.PRIMITIVES:
      op = OPS[primitive](C, stride, affine = False)
      self._ops.append(op)

  def forward(self, x, weights):  #각 연산들에 해당 가중치를을 곱해서 리턴
    return sum(w*op(x) for w, op in zip(weights, self._ops))



class SearchCell(nn.Module):  #셀구성
  def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
    super().__init__()
    self.n_nodes = n_nodes
    if(reduction_p):
      self.preprocess0 = ops.FactorizedReduce(C_pp, C, affine=False)
    else:
      self.preprocess0 = ops.StdConv(C_p, C, 1,1,0,affine=False)

    self.preprocess1 = ops.StdConv(C_p, C, 1,1,0, affine = False)

    self.DAG = nn.ModuleList()
    for i in range(self.n_nodes):
      self.DAG.append(nn.ModuleList())
      for j in range(i+2):
        if reduction and j<2:
          stride = 2
        else:
          stride = 1
        op = ops.MixedOp(C, stride)
        self.DAG[i].append(op)
  def forward(self, s0, s1, w_DAG):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for edges, w_list in zip(self.DAG, w_DAG):
      s_cur = sum(edges[i](s,w) for i, (s,w) in enumerate(zip(states, w_list)))
      states.append(s_cur)

    s_out = torch.cat(states[2:], dim=1)
    return s_out


class SearchCNN(nn.Module):
    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.criterion = criterion

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, C
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def _init_alphas(self):
        """
        initialize architect parameters: alphas
        """
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(self.n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))

    def forward(self, x):
        s0 = s1 = self.stem(x)

        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def weights(self):
        for k, v in self.named_parameters():
            if 'alpha' not in k:
                yield v

    def alphas(self):
        for k, v in self.named_parameters():
            if 'alpha' in k:
                yield v

    def loss(self, X, y):
        logits = self(X)
        return self.criterion(logits, y)



