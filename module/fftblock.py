import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(self, n_head, d_model, d_conv, kernel_size, dropout=0.1):
        super().__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_conv, kernel_size, dropout=dropout)
        
    def forward(self, input_, mask):
        '''
         input_: (batch, max_len, d_model)
         mask  : (batch, max_len)
        '''
        max_len = mask.size(1)
        
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        attn, output = self.slf_attn(input_, mask=slf_attn_mask)
        output = output.masked_fill(mask.unsqueeze(-1), 0)

        output = self.pos_ffn(output)
        output = output.masked_fill(mask.unsqueeze(-1), 0)

        return attn, output


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()
        
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale
        
        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask
        
        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_model, dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.fc_q = nn.Linear(d_model, n_head * self.d_k)
        self.fc_k = nn.Linear(d_model, n_head * self.d_k)
        self.fc_v = nn.Linear(d_model, n_head * self.d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(self.d_k, 0.5))

        self.fc_o = nn.Linear(n_head * self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_x, d_model = x.size()
        n_q = n_k = n_v = n_x

        q = self.fc_q(x) # 1.单头变多头
        k = self.fc_k(x)
        v = self.fc_v(x)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        output = self.dropout(output)
        output = output + x
        output = self.layer_norm(output)

        return attn, output
    

class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_model, d_conv, kernel_size, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(
            d_model,
            d_conv,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )

        self.conv2 = nn.Conv1d(
            d_conv,
            d_model,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        '''
         x: (batch, max_len, d_model)
        '''
        output = x.transpose(1, 2) # (batch, d_model, max_len)
        output = self.conv1(output) # (batch, d_conv, max_len)
        output = F.relu(output) # (batch, d_conv, max_len)
        output = self.conv2(output) # (batch, d_model, max_len)
        output = output.transpose(1, 2) # (batch, max_len, d_model)

        output = self.dropout(output)
        output = output + x
        output = self.layer_norm(output)

        return output
