''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
#import transformer_converter.Constants as Constants
from models.transformer.Layers import TransformerLayer #, DecoderLayer
from tensor_device import *

__author__ = "Copied"

np.random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
mySeed = np.random.RandomState(1234)

# def GetPaddingMask(seq):
#     assert seq.dim() == 2
#     return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

# def GetSinusoidEncodingTable(sentenceLen, hiddenDim, paddingIdx=None):
#     ''' Sinusoid position encoding table '''

#     def calAngle(position, hiddenIdx):
#         return position / np.power(10000, 2 * (hiddenIdx // 2) / hiddenDim)

#     def getPosAngleVec(position):
#         return [calAngle(position, hid_j) for hid_j in range(hiddenDim)]

#     sinusoidTable = np.array([getPosAngleVec(pos_i) for pos_i in range(-sentenceLen,sentenceLen)])

#     sinusoidTable[:, 0::2] = np.sin(sinusoidTable[:, 0::2])  # dim 2i
#     sinusoidTable[:, 1::2] = np.cos(sinusoidTable[:, 1::2])  # dim 2i+1

#     if paddingIdx is not None:
#         # zero vector for padding dimension
#         sinusoidTable[paddingIdx] = 0.

#     return torch.FloatTensor(sinusoidTable).cuda()

# def GetAttentionMask(sequenceK, sequenceQ):
#     ''' For masking out the padding part of key sequence. '''

#     # Expand to fit the shape of key query attention matrix.
#     lengthQ = sequenceQ.size(1)
#     paddingMask = sequenceK.eq(Constants.PAD)   # 0是padding对应的标签数字
#     paddingMask = paddingMask.unsqueeze(1).expand(-1, lengthQ, -1)  # b x lq x lk

#     return paddingMask

# def GetSubsequentMask(seq):
#     ''' For masking out the subsequent info. '''

#     sz_b, len_s = seq.size()
#     subsequent_mask = torch.triu(
#         torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8).cuda(), diagonal=1)
#     subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

#     return subsequent_mask

# class Embedder(nn.Module):

#     def __init__(self,wordEmbed,entityEmbed,relationEmbed,vectorDim,kgDim,maxSentLen,use_cuda):

#         #super().__init__()
#         super(Embedder, self).__init__()
#         sentenceLen = maxSentLen + 1
        
#         self.wordEmb = wordEmbed
        
#         self.relationEmb = relationEmbed
        
#         self.entityEmb = entityEmbed

#         self.positionEmb = nn.Embedding.from_pretrained(
#             GetSinusoidEncodingTable(sentenceLen, vectorDim, paddingIdx=0),
#             freeze=True)
        
#     def forward (self,sentenceWords,sentencePosChem,sentencePosDis,chem,dis,chemLen,disLen,chemical,disease,relation):
#         # -- Prepare masks
#         attentionMask = GetAttentionMask(sequenceK=sentenceWords, sequenceQ=sentenceWords)
#         paddingMask = GetPaddingMask(sentenceWords)

#         # wordEmbedding / relation embedding/ entity emebdding (chemical disease)
#         chemVec      = torch.sum(self.wordEmb(chem),1)/chemLen
#         disVec      = torch.sum(self.wordEmb(dis),1)/disLen
#         #wordsembeddings:
#         sentBatch = []
#         for batch in range(len(sentenceWords)):
#             words = []
#             for word in sentenceWords[batch][0:]:
#                 if word == -1:
#                     words.append(chemVec[batch].unsqueeze(0).unsqueeze(0))
#                 elif word == -2:
#                     words.append(disVec[batch].unsqueeze(0).unsqueeze(0))
#                 else:
#                     words.append(self.wordEmb(word).unsqueeze(0).unsqueeze(0))
#             sentBatch.append(torch.cat(words,1))
#         embeddingWords = torch.cat(sentBatch,0) + self.positionEmb(sentencePosChem) + self.positionEmb(sentencePosDis)
#         embeddingRel = self.relationEmb(relation)
#         embeddingChem  = self.entityEmb(chemical)
#         embeddingDis = self.entityEmb(disease)

#         return embeddingWords,embeddingRel,embeddingChem,embeddingDis, attentionMask, paddingMask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self,layerNum, headNum, vectorDim, entityDim, hiddenDim, dropout=0.1, useAtt =False,use_cuda=True):

        #super().__init__()
        super(Encoder, self).__init__()

        self.useAtt = useAtt
        self.hiddenDim = hiddenDim
        self.dropout = nn.Dropout(0.2)


        self.layerStack = nn.ModuleList([
            TransformerLayer(headNum, vectorDim, entityDim, hiddenDim, dropout=dropout, useAtt =useAtt,use_cuda =use_cuda)
            for _ in range(layerNum)])

        self.converterLayerW = ParameterDevice(mySeed.uniform(-0.01, 0.01, (self.hiddenDim, self.hiddenDim)),use_cuda=use_cuda, requires_grad=True)
        self.converterLayerb = ParameterDevice(mySeed.uniform(-0.01, 0.01, (1,self.hiddenDim)), use_cuda= use_cuda, requires_grad=True)

    def forward(self, encodingOutput, paddingMask,attentionMask, head=None,tail=None, returnAtt=False):

        attentionList = []

        for encLayer in self.layerStack:
            encodingOutput = encLayer(
                encodingOutput,
                paddingMask=paddingMask,
                attentionMask=attentionMask,
                head = head,
                tail = tail)
            #if returnAtt:
            #    attentionList += [attMatrix]
        
        if self.useAtt:
            cleanEncodingVec = F.max_pool1d(encodingOutput.transpose(1,2), encodingOutput.size(1)).transpose(1,2)
            return cleanEncodingVec
        else:
            return encodingOutput

## we do not need decoder

# class Decoder(nn.Module):
#     ''' A decoder model with self attention mechanism. '''

#     def __init__(
#             self,
#             n_tgt_vocab, maxSentLen, vectorDim,
#             n_layers, n_head, d_k, d_v,
#             d_model, d_inner, dropout=0.1):

#         super().__init__()
#         sentenceLen = maxSentLen + 1

#         self.tgt_word_emb = nn.Embedding(
#             n_tgt_vocab, vectorDim, padding_idx=Constants.PAD)

#         self.positionEmb = nn.Embedding.from_pretrained(
#             GetSinusoidEncodingTable(sentenceLen, vectorDim, padding_idx=0),
#             freeze=True)

#         self.layerStack = nn.ModuleList([
#             DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
#             for _ in range(n_layers)])

#     def forward(self, tgt_seq, tgt_pos, src_seq, encodingOutput, returnAtt=False):

#         dec_slf_attn_list, dec_enc_attn_list = [], []

#         # -- Prepare masks
#         paddingMask = GetPaddingMask(tgt_seq)

#         attentionMask_subseq = GetSubsequentMask(tgt_seq)
#         attentionMask_keypad = GetAttentionMask(seq_k=tgt_seq, seq_q=tgt_seq)
#         attentionMask = (attentionMask_keypad + attentionMask_subseq).gt(0)

#         dec_enc_attn_mask = GetAttentionMask(seq_k=src_seq, seq_q=tgt_seq)

#         # -- Forward
#         dec_output = self.tgt_word_emb(tgt_seq) + self.positionEmb(tgt_pos)

#         for dec_layer in self.layerStack:
#             dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
#                 dec_output, encodingOutput,
#                 paddingMask=paddingMask,
#                 attentionMask=attentionMask,
#                 dec_enc_attn_mask=dec_enc_attn_mask)

#             if returnAtt:
#                 dec_slf_attn_list += [dec_slf_attn]
#                 dec_enc_attn_list += [dec_enc_attn]

#         if returnAtt:
#             return dec_output, dec_slf_attn_list, dec_enc_attn_list
#         return dec_output,

# class Transformer(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''

#     def __init__(
#             self,
#             n_src_vocab, n_tgt_vocab, maxSentLen,
#             vectorDim=512, d_model=512, d_inner=2048,
#             n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
#             tgt_emb_prj_weight_sharing=True,
#             emb_src_tgt_weight_sharing=True):

#         super().__init__()

#         self.encoder = Encoder(
#             n_src_vocab=n_src_vocab, maxSentLen=maxSentLen,
#             vectorDim=vectorDim, d_model=d_model, d_inner=d_inner,
#             n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
#             dropout=dropout)

#         self.decoder = Decoder(
#             n_tgt_vocab=n_tgt_vocab, maxSentLen=maxSentLen,
#             vectorDim=vectorDim, d_model=d_model, d_inner=d_inner,
#             n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
#             dropout=dropout)

#         self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)

#         assert d_model == vectorDim, \
#         'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'

#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.

#         if emb_src_tgt_weight_sharing:
#             # Share the weight matrix between source & target word embeddings
#             assert n_src_vocab == n_tgt_vocab, \
#             "To share word embedding table, the vocabulary size of src/tgt shall be the same."
#             self.encoder.wordEmb.weight = self.decoder.tgt_word_emb.weight

#     def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

#         tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

#         encodingOutput, *_ = self.encoder(src_seq, src_pos)
#         dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, encodingOutput)
#         seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

#         return seq_logit.view(-1, seq_logit.size(2))
