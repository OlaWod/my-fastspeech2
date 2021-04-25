import os
import tgt
import json
import random
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import read
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

import myaudio


class Processor:
    def __init__(self, config):
        self.config = config

        self.feature_dir = config['path']['feature_dir']
        self.data_dir = os.path.join(config['path']['feature_dir'], 'data')
        self.tg_dir = os.path.join(config['path']['feature_dir'], 'textgrid')
    
        self.mel_dir = os.path.join(config['path']['feature_dir'], 'mel')
        self.energy_dir = os.path.join(config['path']['feature_dir'], 'energy')
        self.f0_dir = os.path.join(config['path']['feature_dir'], 'f0')
        self.duration_dir = os.path.join(config['path']['feature_dir'], 'duration')

        self.max_wav_value = config['audio']['max_wav_value']
        self.hop_length = config['stft']['hop_length']
        self.sampling_rate = config["audio"]["sampling_rate"]
        self.val_size = config['val_size']

        self.energy_phone_average = config["choice"]["energy_phone_average"]
        self.f0_phone_average = config["choice"]["f0_phone_average"]

        print("Loading STFT...")
        self.stft = myaudio.TacotronSTFT(
            config["stft"]["filter_length"],
            config["stft"]["hop_length"],
            config["stft"]["win_length"],
            config["mel"]["n_mel_channels"],
            config["audio"]["sampling_rate"],
            config["mel"]["mel_fmin"],
            config["mel"]["mel_fmax"],
        )
    
    
    def process_data(self):
        os.makedirs(self.mel_dir, exist_ok=True)
        os.makedirs(self.energy_dir, exist_ok=True)
        os.makedirs(self.f0_dir, exist_ok=True)
        os.makedirs(self.duration_dir, exist_ok=True)

        infos = []
        speakers = {}
        f0_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        print('Start processing...')

        for i, speaker in enumerate(tqdm(os.listdir(self.data_dir))):
            print(speaker)
            speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.data_dir, speaker))):
                if ".wav" not in wav_name:
                    continue
                basename = wav_name.split('.')[0]

                # paths
                wav_path = os.path.join(self.data_dir, speaker, wav_name)
                text_path = os.path.join(self.data_dir, speaker, "{}.lab".format(basename))
                tg_path = os.path.join(self.tg_dir, speaker, "{}.TextGrid".format(basename))
                if not os.path.exists(text_path) or not os.path.exists(tg_path):
                    continue

                # get features
                features = self.get_features(tg_path, wav_path, text_path)
                if features is None:
                    continue
                text, phone, duration, mel, energy, f0 = features

                info = "|".join([basename, speaker, text, phone])
                infos.append(info)
                
                # save
                self.save_npys(duration, mel, energy, f0, basename)

                # for future normalize
                f0 = self.remove_outlier(f0)
                energy = self.remove_outlier(energy)
                if len(f0) > 0:
                    f0_scaler.partial_fit(f0.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

        stats = self.standardize_f0_energy(f0_scaler, energy_scaler)
        
        self.save_infos(infos)
        self.save_speakers(speakers)
        self.save_stats(stats)
        

    def save_npys(self, duration, mel, energy, f0, basename):
        myaudio.save_feature_to_npy(duration, 'duration', self.duration_dir, basename)
        myaudio.save_feature_to_npy(mel.T, 'mel', self.mel_dir, basename)
        myaudio.save_feature_to_npy(energy, 'energy', self.energy_dir, basename)
        myaudio.save_feature_to_npy(f0, 'f0', self.f0_dir, basename)


    def save_infos(self, infos):
        random.shuffle(infos)

        with open(os.path.join(self.feature_dir, "train.txt"), "w", encoding="utf-8") as f:
            for info in infos[self.val_size :]:
                f.write(info + "\n")
        with open(os.path.join(self.feature_dir, "val.txt"), "w", encoding="utf-8") as f:
            for info in infos[: self.val_size]:
                f.write(info + "\n")


    def save_speakers(self, speakers):
        with open(os.path.join(self.feature_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))


    def save_stats(self, stats):
        with open(os.path.join(self.feature_dir, "stats.json"), "w") as f:
            f.write(json.dumps(stats))
            

    def get_features(self, tg_path, wav_path, text_path):
        # text
        with open(text_path, "r") as f:
            text = f.readline().strip("\n")

        # phone, duration, start, end
        phone, duration, start, end = self.process_textgrid(tg_path)
        if start >= end:
            return None

        # wav
        sampling_rate, wav = read(wav_path)
        wav = wav[int(sampling_rate * start) : int(sampling_rate * end)]

        # mel, energy, f0
        mel, energy = myaudio.get_mel_energy_from_wav(wav, sampling_rate, self.stft, self.max_wav_value)
        f0 = myaudio.get_f0_from_wav(wav, sampling_rate, self.hop_length, self.max_wav_value)
        if np.sum(f0 !=0) <= 1:
            return None

        mel = mel[:, :sum(duration)].numpy().astype(np.float32)
        energy = energy[:sum(duration)].numpy().astype(np.float32)
        f0 = f0[: sum(duration)]

        if self.energy_phone_average:
            energy = self.energy_phone_averaging(energy, duration)
        if self.f0_phone_average:
            f0 = self.f0_phone_averaging(f0, duration)

        # return
        return (text, phone, duration, mel, energy, f0)


    def energy_phone_averaging(self, energy, duration):
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                energy[i] = np.mean(energy[pos : pos + d])
            else:
                energy[i] = 0
            pos += d

        return energy[: len(duration)]


    def f0_phone_averaging(self, f0, duration):
        # perform linear interpolation
        nonzero_ids = np.where(f0 != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            f0[nonzero_ids],
            fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]),
            bounds_error=False,
        )
        f0 = interp_fn(np.arange(0, len(f0)))
            
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                f0[i] = np.mean(f0[pos : pos + d])
            else:
                f0[i] = 0
            pos += d

        return f0[: len(duration)]
    

    def process_textgrid(self, tg_path):
        textgrid = tgt.io.read_textgrid(tg_path)
        tier = textgrid.get_tier_by_name("phones")

        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
    
        return "{" + " ".join(phones) + "}", durations, start_time, end_time


    def remove_outlier(self, values):
        # remove too small & too big values
        values = np.array(values)
        p25 = np.percentile(values, 25) # 25% values are < p25
        p75 = np.percentile(values, 75) # 75% values are < p75
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]


    def standardize_f0_energy(self, f0_scaler, energy_scaler):
        f0_mean = f0_scaler.mean_[0]
        f0_std = f0_scaler.scale_[0]
        f0_min, f0_max = self.standardize(
            self.f0_dir, f0_mean, f0_std
        )
        
        energy_mean = energy_scaler.mean_[0]
        energy_std = energy_scaler.scale_[0]
        energy_min, energy_max = self.standardize(
            self.energy_dir, energy_mean, energy_std
        )

        stats = {
            "f0": [
                float(f0_min),
                float(f0_max),
                float(f0_mean),
                float(f0_std),
            ],
            "energy": [
                float(energy_min),
                float(energy_max),
                float(energy_mean),
                float(energy_std),
            ],
        }
        
        return stats

        
    def standardize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filepath = os.path.join(in_dir, filename)
            values = (np.load(filepath) - mean) / std # standardize values
            np.save(filepath, values) # overwrite

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
