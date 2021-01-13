import torch
import torch.nn as nn
import numpy as np

np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
mySeed = np.random.RandomState(1234)

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attDropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attDropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attMatrix = torch.bmm(q, k.transpose(1, 2))
        attMatrix = attMatrix / self.temperature

        if mask is not None:
            attMatrix = attMatrix.masked_fill(mask, -100000.0)

        attMatrix = self.softmax(attMatrix)
        attMatrix = self.dropout(attMatrix)
        output = torch.bmm(attMatrix, v)

        return output#, attMatrix
