''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.transformer.Modules import ScaledDotProductAttention
from tensor_device import *

np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
mySeed = np.random.RandomState(1234)


__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self,headNum, vectorDim, entityDim, hiddenDim, dropout=0.1, useAtt =False,use_cuda = True):
        super().__init__()

        self.headNum = headNum
        self.vectorDim = vectorDim
        self.entityDim = entityDim
        self.hiddenDim = hiddenDim
        #self.useGate = useGate
        
        if useAtt:
            self.Q = ParameterDevice(mySeed.uniform(-0.01, 0.01, (self.vectorDim + self.entityDim * 2 , self.hiddenDim * self.headNum )),use_cuda=use_cuda, requires_grad = True)
        else:
            self.Q = ParameterDevice(mySeed.uniform(-0.01, 0.01, (self.vectorDim, self.hiddenDim * self.headNum )),use_cuda=use_cuda, requires_grad = True)
        self.K = ParameterDevice(mySeed.uniform(-0.01, 0.01, (self.vectorDim, self.hiddenDim * self.headNum )),use_cuda=use_cuda, requires_grad = True)
        self.V = ParameterDevice(mySeed.uniform(-0.01, 0.01, (self.vectorDim, self.hiddenDim * self.headNum )),use_cuda=use_cuda, requires_grad = True)
        self.O = ParameterDevice(mySeed.uniform(-0.01, 0.01, (self.hiddenDim * self.headNum, self.hiddenDim )),use_cuda =use_cuda, requires_grad = True)
        
        # if useGate:
        #     self.gateW1 = ParameterDevice (mySeed.uniform(-0.01, 0.01, (self.kgDim,1)),use_cuda =use_cuda, requires_grad = True)
        #     self.gateW2 = ParameterDevice (mySeed.uniform(-0.01, 0.01, (self.kgDim,1)),use_cuda=use_cuda, requires_grad = True)
        #     self.gateW3 = ParameterDevice (mySeed.uniform(-0.01, 0.01, (self.hiddenDim,1)),use_cuda =use_cuda, requires_grad = True)
        #     self.gateb  = ParameterDevice (mySeed.uniform(-0.01, 0.01, (1,1)),use_cuda= use_cuda, requires_grad = True)

        # self.w_qs = nn.Linear(d_model, n_head * d_k)
        # self.w_ks = nn.Linear(d_model, n_head * d_k)
        # self.w_vs = nn.Linear(d_model, n_head * d_v)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.hiddenDim, 0.5))
        self.layerNorm = nn.LayerNorm(self.vectorDim).cuda()

        # self.fc = nn.Linear(n_head * d_v, d_model)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout).cuda()


    def forward(self, q, k, v, mask=None):

        batchSize, sentLength, _ = q.size()

        #residual = v  #源代码是q，这里可以考虑一下。
        
        q = torch.einsum('ijk,kp->ijp',q,self.Q).view(batchSize,sentLength,self.headNum,self.hiddenDim)  #与bmm 即batch点乘等价
        k = torch.einsum('ijk,kp->ijp',k,self.K).view(batchSize,sentLength,self.headNum,self.hiddenDim)
        v = torch.einsum('ijk,kp->ijp',v,self.V).view(batchSize,sentLength,self.headNum,self.hiddenDim)   #batch * length * (headNum * hiddenDim)
        
        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, sentLength, self.hiddenDim) # (n*b) x length x hiddenDim
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, sentLength, self.hiddenDim)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, sentLength, self.hiddenDim)

        mask = mask.repeat(self.headNum, 1, 1) # (n*b) x .. x ..
        output = self.attention(q, k, v, mask=mask)

        output = output.view(self.headNum, batchSize, sentLength, self.hiddenDim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batchSize, sentLength, -1) # b x lq x (n*dv)
        output = self.dropout(torch.einsum('ijk,kp->ijp',output,self.O))
        
        # if self.useGate:
        #     gateR = torch.mm(relation,self.gateW1).unsqueeze(1).expand(-1,sentLength,-1)
        #     gateE = torch.mm(disease-chemical,self.gateW2).unsqueeze(1).expand(-1,sentLength,-1)
        #     gateC = torch.einsum('ijk,kp->ijp',output,self.gateW3)
        #     gate = torch.sigmoid (gateR + gateE + gateC + self.gateb) # 
        #     #gate = torch.sigmoid (self.cos(relation,e))
        #     output = gate * output + (1-gate) * residual
        # else:
        #print (output.size(),residual.size())
        output = self.layerNorm(output)# + residual)

        return output #, attMatrix

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, hiddenDim, dropout=0.1,use_cuda =True):
        super().__init__()
        self.hiddenDim = hiddenDim
        
        self.feedForward1 = nn.Conv1d(self.hiddenDim, self.hiddenDim, 1).cuda() # position-wise
        self.feedForward2 = nn.Conv1d(self.hiddenDim, self.hiddenDim, 1).cuda() # position-wise
        self.layerNorm = nn.LayerNorm(self.hiddenDim).cuda()
        self.dropout = nn.Dropout(dropout).cuda()

    def forward(self, context):
        residual = context
        output = context.transpose(1, 2)
        output = self.feedForward2(F.relu(self.feedForward1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layerNorm(output + residual)
        return output
