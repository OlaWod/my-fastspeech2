import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict

from .layers import Conv1D
from utils import pad, get_mask_from_lengths


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_cfg, model_cfg):
        super().__init__()
        self.duration_predictor = VariancePredictor(model_cfg)
        self.pitch_predictor = VariancePredictor(model_cfg)
        self.energy_predictor = VariancePredictor(model_cfg)
        self.length_regulator = LengthRegulator()

        n_bins = model_cfg["variance_embedding"]["n_bins"]

        with open(
            os.path.join(preprocess_cfg["path"]["feature_dir"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["f0"][:2]
            energy_min, energy_max = stats["energy"][:2]
        
        self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_cfg["encoder"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_cfg["encoder"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, mask, truth):
        prediction = self.pitch_predictor(x, mask) # (batch, max_src_len)

        if truth is not None: # for training
            embedding = self.pitch_embedding(
                    torch.bucketize(truth, self.pitch_bins)
            ) # (batch, max_src_len, d_model)
        else: # for inference
            embedding = self.pitch_embedding(
                    torch.bucketize(prediction, self.pitch_bins)
            ) # (batch, max_src_len, d_model)
        return prediction, embedding

    def get_energy_embedding(self, x, mask, truth):
        prediction = self.energy_predictor(x, mask) # (batch, max_src_len)

        if truth is not None: # for training
            embedding = self.energy_embedding(
                    torch.bucketize(truth, self.energy_bins)
            ) # (batch, max_src_len, d_model)
        else: # for inference
            embedding = self.energy_embedding(
                    torch.bucketize(prediction, self.energy_bins)
            ) # (batch, max_src_len, d_model)
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_mel_len=None,
        pitch_truth=None,
        energy_truth=None,
        duration_truth=None,
    ):
        '''
         x: (batch, max_src_len, d_model)
         src_mask: (batch, max_src_len)
         max_mel_len: 1
        '''
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, src_mask, pitch_truth
        )
        x = x + pitch_embedding # (batch, max_src_len, d_model)
            
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, src_mask, energy_truth
        )
        x = x + energy_embedding # (batch, max_src_len, d_model)

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_truth is not None: # for training
            x, mel_len = self.length_regulator(x, duration_truth, max_mel_len) # (batch, max_mel_len, d_model), (batch)
        else: # for inference
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) ),
                min=0,
            ) # (batch, max_src_len)
            x, mel_len = self.length_regulator(x, duration_rounded) # (batch, max_mel_len, d_model), (batch)
            mel_mask = get_mask_from_lengths(mel_len)

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super().__init__()

    def expand(self, batch, dur):
        '''
         batch: (max_src_len, d_model)
         dur: (max_src_len)
        '''
        out = list()

        for i, vec in enumerate(batch):
            expand_size = dur[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1)) # (dur[i], d_model)
        out = torch.cat(out, 0) # (sum(dur[i]=pred_mel_len, d_model))

        return out

    def forward(self, x, duration, max_mel_len=None):
        '''
         x: (batch, max_src_len, d_model)
         duration: (batch, max_src_len)
         max_mel_len: 1
        '''
        output = list()
        mel_len = list()
        for batch, dur in zip(x, duration):
            expanded = self.expand(batch, dur) # (pred_mel_len, d_model)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        output = pad(output, max_mel_len) # (batch, max_mel_len, d_model)
        mel_len = torch.LongTensor(mel_len).to(x.device) # (batch)

        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.d_model = model_config["encoder"]["encoder_hidden"]
        self.filter_n = model_config["variance_predictor"]["filter_n"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv1D(
                            self.d_model,
                            self.filter_n,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_n)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv1D(
                            self.filter_n,
                            self.filter_n,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_n)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                    ("linear", nn.Linear(self.filter_n, 1))
                ]
            )
        )

    def forward(self, enc_output, mask):
        '''
         enc_output: (batch, max_src_len, d_model)
         mask: (batch, max_src_len)
        '''
        output = self.layers(enc_output) # (batch, max_src_len, 1)
        output = output.squeeze(-1) # (batch, max_src_len)

        if mask is not None:
            output = output.masked_fill(mask, 0.0)

        return output
