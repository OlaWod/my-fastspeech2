import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from scipy.io import wavfile
import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

from dataset import TextDataset
from myvocoder import get_vocoder, vocoder_infer
from model import get_model
from utils import to_device


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    preprocess_cfg, model_cfg, train_cfg = configs

    # dataset
    dataset = TextDataset(args.source, preprocess_cfg)
    loader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collate_fn,
    )
    # model
    model = get_model(args, configs, device, train=False)
    # vocoder
    vocoder = get_vocoder("HiFi-GAN", device)

    # synthesize
    for batch in loader:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(*batch)
        #output = (2333, torch.rand(1, 200, 80))
        synth(batch, output, vocoder, configs)
        

def synth(batch, output, vocoder, configs):
    preprocess_cfg, model_cfg, train_cfg = configs
    sr = preprocess_cfg["audio"]["sampling_rate"]
    result_dir = train_cfg["path"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    
    basenames = batch[0]
    mel_lens = output[-1]
    mels = output[1].transpose(1, 2) # (batch, 80, mel_len)
    #wavs = vocoder_infer(vocoder, mels, "HiFi-GAN")
    
    for i, (basename, mel) in enumerate(zip(basenames, mels)):
        mel = mel[:,:mel_lens[i]]
        wav = vocoder_infer(vocoder, mel.unsqueeze(0), "HiFi-GAN")[0]
        mel = mel.cpu().numpy().astype(np.float32)
        plt.imshow(mel)
        plt.ylim(0, mel.shape[0])
        plt.colorbar()
        plt.savefig(os.path.join(result_dir, "{}.png".format(basename)))
        plt.close()
        
        wavfile.write(os.path.join(result_dir,"{}.wav".format(basename)), sr, wav)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="path to source file",
    )
    parser.add_argument(
        "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Load Config
    preprocess_cfg = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_cfg = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_cfg = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_cfg, model_cfg, train_cfg)

    # synthesize
    main(args, configs)
    
    # model
    model = get_model(args, configs, device, train=False)
    
