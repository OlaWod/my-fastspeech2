import os
import json
import torch
import torch.nn as nn

from module import Encoder, Decoder, VarianceAdaptor, PostNet
from utils import get_mask_from_lengths


class MyModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        preprocess_cfg, model_cfg, train_cfg = configs
        
        self.encoder = Encoder(model_cfg)
        self.variance_adaptor = VarianceAdaptor(preprocess_cfg, model_cfg)
        self.decoder = Decoder(model_cfg)
        self.mel_linear = nn.Linear(
            model_cfg["decoder"]["decoder_hidden"],
            preprocess_cfg["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_cfg["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_cfg["path"]["feature_dir"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_cfg["encoder"]["encoder_hidden"],
            )

    def forward(
        self,

        basenames,
        speaker_ids,
        phone_ids,

        src_lens,
        max_src_len,
        mel_lens=None,
        max_mel_len=None,

        mels=None,
        energys=None,
        f0s=None,
        durations=None,
    ):
        '''
         basenames: (batch), list
         speaker_ids: (batch), numpy.ndarray
         phone_ids: (batch, max_src_len), numpy.ndarray

         src_lens: (batch), numpy.ndarray
         max_src_len: 1, numpy.int64
         mel_lens: (batch), numpy.ndarray
         max_mel_len: 1, numpy.int64

         mels: (batch, max_mel_len, 80), numpy.ndarray
         energys: (batch, max_src_len), numpy.ndarray
         f0s: (batch, max_src_len), numpy.ndarray
         durations: (batch, max_src_len), numpy.ndarray
        '''
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len) # (batch, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        ) # (batch, max_mel_len)

        output = self.encoder(phone_ids, src_masks) # (batch, max_src_len, d_model)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speaker_ids).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            f0s,
            energys,
            durations,
        )

        # output: (batch, max_mel_len, d_model)
        # mel_masks: (batch, max_mel_len)
        output = self.decoder(output, mel_masks) # (batch, max_mel_len, d_model)
        output = self.mel_linear(output) # (batch, max_mel_len, 80)

        postnet_output = self.postnet(output) + output # (batch, max_mel_len, 80)

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            src_masks,
            mel_masks,
            mel_lens
        )
