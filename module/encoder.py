import torch
import torch.nn as nn
import numpy as np

from .fftblock import FFTBlock
from mytext import symbols
from utils import get_sinusoid_encoding_table


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_src_vocab = len(symbols) + 1
        n_position = config["max_seq_len"] + 1
        
        n_layers = config["encoder"]["encoder_layer"]
        n_head = config["encoder"]["encoder_head"]
        d_model = config["encoder"]["encoder_hidden"]
        dropout = config["encoder"]["encoder_dropout"]

        d_conv = config["fftblock"]["conv_filter_size"]
        kernel_size = config["fftblock"]["conv_kernel_size"]
        
        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model
        
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_model, padding_idx=0
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_model).unsqueeze(0),
            requires_grad=False,
        ) # (1, max_seq_len, d_model)
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    n_head, d_model, d_conv, kernel_size, dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask):
        '''
         src_seq: (batch, max_len)
         mask: (batch, max_len)
        '''
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Forward
        if max_len > self.max_seq_len:
            position_enc = nn.Parameter(
                get_sinusoid_encoding_table(max_len, self.d_model).unsqueeze(0),
                requires_grad=False,
            )
            output = self.src_word_emb(src_seq) + position_enc[
                :, :max_len, :].expand(batch_size, -1, -1).to(src_seq.device)
        else:
            output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :].expand(batch_size, -1, -1)

        for fftblock in self.layer_stack:
            attn, output = fftblock(output, mask)

        return output
