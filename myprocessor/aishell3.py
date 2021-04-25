import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def prepare_data(config):
    in_dir = config["path"]["corpus_dir"]
    out_dir = os.path.join(config["path"]["feature_dir"], "data")
    
    sampling_rate = config["audio"]["sampling_rate"]
    max_wav_value = config["audio"]["max_wav_value"]
    
    for subdataset in ["train", "test"]:
        print("Processing {}ing set...".format(subdataset))
        with open(os.path.join(in_dir, subdataset, "content.txt"), encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                wav_name, text = line.strip("\n").split("\t")
                speaker = wav_name[:7]
                text = text.split(" ")[1::2]
                
                wav_path = os.path.join(in_dir, subdataset, "wav", speaker, wav_name)
                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)

                    # copy and normalize basename.wav
                    if config["choice"]["norm_wav"]:
                        wav, _ = librosa.load(wav_path, sampling_rate)
                        wav = wav / max(abs(wav)) * max_wav_value
                        wavfile.write(
                            os.path.join(out_dir, speaker, wav_name),
                            sampling_rate,
                            wav.astype(np.int16),
                        )

                    # write basename.lab
                    with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(wav_name[:11])),
                        "w",
                    ) as f1:
                        f1.write(" ".join(text))
                        
