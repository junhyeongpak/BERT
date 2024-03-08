import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False # 그레디언트를 계산해야하는지 여부

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_tem = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_tem)
        pe[:, 1::2] = torch.cos(position * div_tem)

        pe = pe.unsqueeze(0)
        
        # pe -? [0, max_len, d_model]

        # 등록된 버퍼는 'state_dict'에 포함되어 저장되고 다시 불러올 수 있다.
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]