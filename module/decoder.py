import torch
import torch.nn as nn
import numpy as np

from .fftblock import FFTBlock
from utils import get_sinusoid_encoding_table


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        
        n_layers = config["decoder"]["decoder_layer"]
        n_head = config["decoder"]["decoder_head"]
        d_model = config["decoder"]["decoder_hidden"]
        dropout = config["decoder"]["decoder_dropout"]
        
        d_conv = config["fftblock"]["conv_filter_size"]
        kernel_size = config["fftblock"]["conv_kernel_size"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

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

    def forward(self, enc_seq, mask):
        '''
         enc_seq: (batch, max_mel_len, d_model)
         mask: (batch, max_mel_len)
        '''
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if max_len > self.max_seq_len:
            position_enc = nn.Parameter(
                get_sinusoid_encoding_table(max_len, self.d_model).unsqueeze(0),
                requires_grad=False,
            )
            output = enc_seq + position_enc[
                :, :max_len, :].expand(batch_size, -1, -1).to(enc_seq.device)
        else:
            output = enc_seq + self.position_enc[
                :, :max_len, :].expand(batch_size, -1, -1)

        for fftblock in self.layer_stack:
            attn, output = fftblock(output, mask)

        return output
    
