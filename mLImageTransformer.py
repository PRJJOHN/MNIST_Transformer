import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
N_DIV = 1
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
    def __init__(self, in_channels, size,
                 nfeatures, nclasses, nheads, dropout):
        super().__init__()
        self.b_conv = nn.Sequential(
            nn.Conv2d(in_channels, 15, kernel_size=3, padding=1,stride = 1),
            nn.ReLU()
        )
        self.s_conv = nn.Sequential(
            nn.Conv2d(15, 30, kernel_size=3, padding=1,stride = 2),
            nn.ReLU()
        )
        self.t_conv = nn.Sequential(
            nn.Conv2d(30, 45, kernel_size=3, padding=1,stride = 2),
            nn.ReLU()
        )
        self.f_conv = nn.Sequential(
            nn.Conv2d(45, 60, kernel_size=3, padding=1,stride = 2),
            nn.ReLU()
        )
        self.ff_conv = nn.Sequential(
            nn.Conv2d(60, nfeatures, kernel_size=3, padding=1,stride = 2),
            nn.ReLU()
        )
        attn = MultiHeadedAttention(h=nheads, d_model=nfeatures, dropout=dropout)
        ff = PositionwiseFeedForward(nfeatures, d_ff=2*nfeatures, dropout=dropout)
        self.attn = Encoder(EncoderLayer(nfeatures, attn, ff, dropout), 1)
        self.cls = nn.Linear(4, nclasses)
        #self.cls = nn.Linear(nfeatures, nclasses)
        self.sparse_trans = nn.Linear(size[0], nfeatures)
        self.nfeatures = nfeatures

        # self.init_(self.conv)
        self.init_(self.attn)
        # self.init_(self.cls)

    def init_(self, module):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.b_conv(x)
        x = self.s_conv(x)
        x = self.t_conv(x)
        x = self.f_conv(x)
        x = self.ff_conv(x)
        x = x.reshape(x.size(0), self.nfeatures, -1).permute(0, 2, 1)
        #x = x.view(-1, 28, 28)
        #x = self.sparse_trans(x)
        x = self.attn(x, None)
        #nail = x[:,-1,:]        
        #x = self.cls(nail)
        x = torch.mean(x, dim=-1)
        x = self.cls(x)
        return x