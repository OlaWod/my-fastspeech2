import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from mytext import _clean_text


def prepare_data(config):
    in_dir = config["path"]["corpus_dir"]
    out_dir = os.path.join(config["path"]["feature_dir"], "data")

    sampling_rate = config["audio"]["sampling_rate"]
    max_wav_value = config["audio"]["max_wav_value"]
    cleaners = config["text"]["text_cleaner"]

    speaker = "LJ"

    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        print("Start processing...")
        for line in tqdm(f.readlines()):
            parts = line.strip().split("|")
            basename = parts[0]
            text = parts[2]
            text = _clean_text(text, cleaners)
            
            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(basename))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)

                # copy and normalize basename.wav
                if config["choice"]["norm_wav"]:                
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}.wav".format(basename)),
                        sampling_rate,
                        wav.astype(np.int16),
                    )

                # write basename.lab
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(basename)),
                    "w",
                ) as f1:
                    f1.write(text)
