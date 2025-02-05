import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        if self.weight is not None:
            sample_weights = torch.zeros_like(target, dtype=torch.float32)
            sample_weights[target == 0] = self.weight[0]
            sample_weights[target == 1] = self.weight[1]
        else:
            sample_weights = None
        
        return focal_loss(F.binary_cross_entropy(input, target, reduction='none', weight=sample_weights), self.gamma)

def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, num_classes, weight=None, alpha=10000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)),1) # N * 1
        ib = grads*features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.binary_cross_entropy(input, target, reduction='none', weight=self.weight), ib)



class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.8, weight=None, s=80, gpu=0):
        super(LDAMLoss, self).__init__()
        self.gpu = gpu
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda(self.gpu)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):

        N = x.size(0)
        C = self.m_list.size(0)  

        index = torch.zeros(N, C, dtype=torch.bool).cuda(self.gpu)

        target = target.view(-1, 1)
        target = target.to(torch.int64)
        index.scatter_(1, target, 1)
        index_float = index.type(torch.FloatTensor).cuda(self.gpu)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view(-1, 1)

        x = x.unsqueeze(1).expand(-1, C)
        x_m = x - batch_m
        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s * output, target.view(-1), weight=self.weight)


class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target, use_multiplicative=True):
        output = x / self.Delta_list + self.iota_list if use_multiplicative else x + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)