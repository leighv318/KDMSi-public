import pdb
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn as nn
from torch.autograd import Variable
import math

'''
    单纯的位置编码
'''
class PositionalEncoding(nn.Module):
    def __init__(self, dim, DEVICE, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim, device=DEVICE)
        position = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,dim,2,device=DEVICE) * (-math.log(10000.0) / dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 
        self.register_buffer('pe',pe)

    def forward(self, x):
        
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return x

'''
    位置编码，但是把时间步长的信息加进去了
'''
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
        # pdb.set_trace()
        # print(x.shape,'\t',t)
        # print()
        x = x + Variable(self.pe[:, t*x.size(1) : (t+1)*x.size(1)],requires_grad=False)
        return x
    

'''
    输入 b,c,h,w 进行squeeze and extension的通道注意力机制
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y.expand_as(x)
    
'''
    输入一个列表，每个元素大小为 b,xi，其中xi不固定，表示某一种类型的特征
    进行特征类型的注意力
    channel=特征类数
'''
class SELayer_Mutation(nn.Module):
    def __init__(self, reduction='one', splices=None):
        super(SELayer_Mutation, self).__init__()
        self.splices = splices

        if reduction == 'one':
            self.linears_1 = nn.ModuleList(
                [nn.Linear(splices[i+1]-splices[i], 1, bias=False) for i in range(len(splices)-1)]
            )
            self.linears_2 = nn.Sequential(
                nn.Linear(len(splices)-1, len(splices)-1, bias=False),
                nn.ReLU(),
            )
            self.linears_3 = nn.ModuleList(
                [nn.Linear(1, splices[i+1]-splices[i], bias=False) for i in range(len(splices)-1)]
            )
        # elif reduction == 'same':
        #     self.linears_1 = nn.ModuleList(
        #         [nn.Linear(splices[i+1]-splices[i], splices[i+1]-splices[i], bias=False) for i in range(len(splices)-1)]
        #     )
        #     self.linears_2 = nn.Sequential(
        #         nn.Linear(splices[-1], splices[-1], bias=False),
        #         nn.ReLU(),
        #     )
        #     self.linears_3 = nn.ModuleList(
        #         [nn.Linear(splices[i+1]-splices[i], splices[i+1]-splices[i], bias=False) for i in range(len(splices)-1)]
        #     )
        else:
            raise ValueError('unexpected %s'%reduction)
        
        self.sig = nn.Sigmoid()
        


    def forward(self, x):

        y = []
        for i in range(len(self.splices)-1):
            y.append(self.linears_1[i](x[:, self.splices[i]:self.splices[i+1]]))
        y = torch.cat(y, dim=1) 

        att = self.linears_2(y)

        out = []
        for i in range(len(self.splices)-1):
            out.append(self.linears_3[i](att[:,i].unsqueeze(1)))
        out = torch.cat(out, dim=1) 

        return self.sig(out) * x
        
        