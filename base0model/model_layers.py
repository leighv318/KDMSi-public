"""Dual Attention Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn import init
from torch.autograd import Variable
import math

'''Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu.
        "Dual Attention Network for Scene Segmentation." *CVPR*, 2019'''

class _PositionAttentionModule(nn.Module):
    """Length-agnostic, learnable position attention.

    Input/Output: [batch, length, dim].
    This keeps the original idea of learnable Q/K/V projections, but projects
    along the feature dimension rather than treating sequence length as Conv2d
    channels. Therefore it works with fixed padding length 25 and does not
    depend on hard-coded length 23.
    """

    def __init__(self, in_channels=None, dim=None, reduction=8, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        if dim is None:
            # In this project initial_map maps features to 16 dimensions.
            # Keeping a default avoids changing model.py constructor calls.
            dim = 16
        hidden = max(1, dim // reduction)
        self.query = nn.Linear(dim, hidden, bias=False)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, length, dim]
        q = self.query(x)                    # [B, L, hidden]
        k = self.key(x)                      # [B, L, hidden]
        v = self.value(x)                    # [B, L, dim]
        scale = math.sqrt(max(q.size(-1), 1))
        attention = torch.bmm(q, k.transpose(1, 2)) / scale
        attention = self.softmax(attention)  # [B, L, L]
        out = torch.bmm(attention, v)        # [B, L, dim]
        return self.alpha * out + x


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        pdb.set_trace()
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = feat_a.permute(0, 2, 1) # b, h*w, c
        attention = torch.bmm(feat_a, feat_a_transpose) # b, c, c
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention # b, c, 1
        attention = self.softmax(attention_new) # b, c, c

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out
    

'''  卷积模块 ------------------------------------------------------------ '''

class SequenceConv(nn.Module):
    # 1D convolution over sequence features. Can either return flattened per-position
    # features (legacy behavior) or pooled fixed-length features (variable-length safe).
    def __init__(self, kernel_size=list(range(5,24)), kernel_dim=28, n=0, padding_shape=0, tem=24, pool=False):
        super(SequenceConv, self).__init__()
        self.kernel_size = list(kernel_size)
        self.pool = pool
        self.conv1ds = nn.ModuleList(
            [nn.Conv1d(in_channels=kernel_dim, out_channels=1, kernel_size=i, padding=padding_shape,
                       padding_mode='replicate') for i in self.kernel_size])
        self.relus = nn.ModuleList([nn.ReLU() for i in self.kernel_size])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def forward(self, inputs): # b,l,d
        inputs = inputs.permute(0,2,1) # b,d,l
        out = []
        seq_len = inputs.size(-1)
        for i, conv in enumerate(self.conv1ds):
            k = self.kernel_size[i]
            # If a very short sequence appears, skip invalid convolution and emit 0.
            if seq_len < k:
                pooled = inputs.new_zeros(inputs.size(0), 1)
                out.append(pooled if self.pool else inputs.new_zeros(inputs.size(0), 0))
                continue
            relu_out = self.relus[i](conv(inputs))  # b,1,L-k+1
            if self.pool:
                # Fixed one feature per kernel, independent of sequence length.
                out.append(torch.amax(relu_out, dim=2))  # b,1
            else:
                out.append(relu_out.squeeze(1))

        return torch.cat(out, dim=1) if len(out) > 0 else inputs.new_zeros(inputs.size(0), 0)



class PositionalEncodingWithTime(nn.Module):
    def __init__(self, dim, DEVICE, dropout, max_len):
        super(PositionalEncodingWithTime, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim, device=DEVICE)
        position = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,dim,2,device=DEVICE) * (-math.log(10000.0) / dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 
        self.register_buffer('pe',pe)

    def forward(self, x, t):
        # x: [length, batch, dim]  (model.py 里 z = x.permute((1,0,2)) 之后传进来)
        # self.pe: [1, max_len, dim]
        # 注意: 必须用真实序列长度 x.size(0) 来切 pe, 而不是 x.size(1)(那是 batch)。
        # 旧写法用 x.size(1)=batch 去切, batch 大时 (t*batch) 会越过 max_len 报错,
        # 且即便不越界, 加的也是错位的位置编码。
        L = x.size(0)
        pos = self.pe[:, t * L : (t + 1) * L, :]   # [1, L, dim]
        pos = pos.permute(1, 0, 2)                  # [L, 1, dim], 广播到 [L, batch, dim]
        x = x + Variable(pos, requires_grad=False)
        return x
    
'''
    输入 b,c,h,w 进行squeeze and extension的通道注意力机制
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y.expand_as(x)