import os
import torch
import numpy as np
from torch.nn import functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if len(data) == 5:
        (
            basenames,
            speaker_ids,
            phone_ids,

            phoneid_lens,
            max_phoneid_lens,
        ) = data

        speaker_ids = torch.from_numpy(speaker_ids).long().to(device)
        phone_ids = torch.from_numpy(phone_ids).long().to(device)
        phoneid_lens = torch.from_numpy(phoneid_lens).to(device)

        return (
            basenames,
            speaker_ids,
            phone_ids,

            phoneid_lens,
            max_phoneid_lens,
        )
        
    (
        basenames,
        speaker_ids,
        phone_ids,

        phoneid_lens,
        max_phoneid_lens,
        mel_lens,
        max_mel_lens,

        mels,
        energys,
        f0s,
        durations,
    ) = data

    speaker_ids = torch.from_numpy(speaker_ids).long().to(device)
    phone_ids = torch.from_numpy(phone_ids).long().to(device)
    mels = torch.from_numpy(mels).float().to(device)
    energys = torch.from_numpy(energys).float().to(device)
    f0s = torch.from_numpy(f0s).float().to(device)
    durations = torch.from_numpy(durations).long().to(device)
    phoneid_lens = torch.from_numpy(phoneid_lens).to(device)
    mel_lens = torch.from_numpy(mel_lens).to(device)

    return (
        basenames,
        speaker_ids,
        phone_ids,

        phoneid_lens,
        max_phoneid_lens,
        mel_lens,
        max_mel_lens,

        mels,
        energys,
        f0s,
        durations,
    )


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def pad_1D(inputs):
    def pad(x, max_len, PAD=0):     
        x_padded = np.pad(
            x, (0, max_len - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad(x, max_len) for x in inputs])

    return padded


def pad_2D(inputs):
    def pad(x, max_len, PAD=0):
        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:,:s]

    max_len = max(np.shape(x)[0] for x in inputs)
    padded = np.stack([pad(x, max_len) for x in inputs])

    return padded


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask # (batch, max_len)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table) # (n_pos, d_hid)


def pad(inputs, max_mel_len=None):
    '''
     inputs: [(pred_mel_len, d_model)]
     max_mel_len: 1
    '''
    max_len = max([inputs[i].size(0) for i in range(len(inputs))])
    if max_mel_len and max_mel_len > max_len:
        max_len = max_mel_len

    out_list = list()
    for i, batch in enumerate(inputs):
        one_batch_padded = F.pad(
            batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
        ) # (max_len, d_model)
        out_list.append(one_batch_padded)
        
    out_padded = torch.stack(out_list) # (batch, max_len, d_model)

    return out_padded
