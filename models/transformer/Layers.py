''' Define the Layers '''
import torch
import torch.nn as nn
import numpy as np
from models.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
mySeed = np.random.RandomState(1234)

class TransformerLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, headNum, vectorDim, entityDim, hiddenDim, dropout=0.1,useAtt=False,use_cuda=True):
        super(TransformerLayer, self).__init__()
        self.useAtt = useAtt
        self.selfAttention = MultiHeadAttention(headNum, vectorDim, entityDim, hiddenDim, dropout=dropout,useAtt=useAtt,use_cuda =use_cuda)
        self.posFeedForward = PositionwiseFeedForward(hiddenDim, dropout=dropout,use_cuda =use_cuda)

    def forward(self, contextWords, paddingMask, attentionMask,head =None,tail = None):
        sentenceLength = contextWords.size(1)
        if self.useAtt:
            Head = head.unsqueeze(1).expand(-1,sentenceLength,-1)
            Tail = tail.unsqueeze(1).expand(-1,sentenceLength,-1)
            contextWordswithEnt = torch.cat([contextWords,Head,Tail],2)
        else:
            contextWordswithEnt = contextWords

        contextOutput = self.selfAttention(contextWordswithEnt, contextWords, contextWords, mask=attentionMask)
        contextOutput *= paddingMask

        contextOutput = self.posFeedForward(contextOutput)
        contextOutput *= paddingMask

        return contextOutput

