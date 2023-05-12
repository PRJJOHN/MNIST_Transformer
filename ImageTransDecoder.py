import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
N_DIV = 2
def clones(module: nn.Module, 
           N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) 
                          for _ in range(N)])
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [ l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
              for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = self.get_score(query, key)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def get_score(self, query, key):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        return scores
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
class ImageTransformer(nn.Module):
    def __init__(self, size,
                 nfeatures, nclasses, nheads, dropout):
        super().__init__()
        attn = MultiHeadedAttention(h=nheads, d_model=nfeatures, dropout=dropout)
        ff = PositionwiseFeedForward(nfeatures, d_ff=2*nfeatures, dropout=dropout)
        self.attn = Encoder(EncoderLayer(nfeatures, attn, ff, dropout), 2)
        self.sparse_trans = nn.Linear(size[0], nfeatures)
        self.cls = nn.Linear(nfeatures, size[0])
        attn2 = MultiHeadedAttention(h=nheads, d_model=nfeatures, dropout=dropout)
        ff2= PositionwiseFeedForward(nfeatures, d_ff=2*nfeatures, dropout=dropout)
        self.attn2 = Encoder(EncoderLayer(nfeatures, attn2, ff2, dropout), 2)
        self.sparse_trans2 = nn.Linear(size[0], nfeatures)
        self.cls2 = nn.Linear(nfeatures, size[0])
        self.sigmoid = nn.Sigmoid()
        self.nfeatures = nfeatures

        # self.init_(self.conv)
        self.init_(self.attn)
        self.init_(self.sparse_trans)
        self.init_(self.cls)
        self.init_(self.attn2)
        self.init_(self.sparse_trans2)
        self.init_(self.cls2)
    def init_(self, module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x = self.conv(x)
        # x = x.reshape(x.size(0), self.nfeatures, -1).permute(0, 2, 1)
        x = x.view(-1, 28, 28)
        x2 = x.transpose(1,2)
        x = self.sparse_trans(x)
        x2 = self.sparse_trans2(x2)
        if x.is_cuda:
            mask = torch.ones_like(torch.ones(28,28)).fill_diagonal_(0).cuda()
        else:
            mask = torch.ones_like(torch.ones(28,28)).fill_diagonal_(0)
        mask.unsqueeze_(0)
        x = self.attn(x, mask)
        x2 = self.attn2(x2, mask)
        x = self.cls(x)
        x2 = self.cls2(x2)
        x2 = x2.transpose(1,2)
        #nail = x[:,-1,:]        
        #x = self.cls(nail)
        # x = torch.mean(x, dim=-1)
        out = self.sigmoid(x+x2)
        return out