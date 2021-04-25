import os
import json
import numpy as np
from torch.utils.data import Dataset

from mytext import phone_to_sequence, process_english, process_mandarin
from utils import pad_1D, pad_2D


class MyDataset(Dataset):
    def __init__(self, metafile, preprocess_cfg, train_cfg, sort=False, drop_last=False):
        self.feature_dir = preprocess_cfg['path']['feature_dir']
        self.cleaner = preprocess_cfg['text']['text_cleaner']
        self.basename, self.speaker, self.text, self.phone = self.process_meta(metafile)
        with open(os.path.join(self.feature_dir, "speakers.json")) as f:
            self.speaker_map = json.load(f)

        self.batch_size = train_cfg['batch_size']
        self.sort = sort
        self.drop_last = drop_last
        
    
    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        text = self.text[idx]
        phone = self.phone[idx]
        phone_id = np.array(phone_to_sequence(phone, self.cleaner))

        # mel
        mel_path = os.path.join(self.feature_dir, 'mel', 'mel-{}.npy'.format(basename))
        mel = np.load(mel_path)
        # energy
        energy_path = os.path.join(self.feature_dir, 'energy', 'energy-{}.npy'.format(basename))
        energy = np.load(energy_path)
        # f0
        f0_path = os.path.join(self.feature_dir, 'f0', 'f0-{}.npy'.format(basename))
        f0 = np.load(f0_path)
        # duration
        duration_path = os.path.join(self.feature_dir, 'duration', 'duration-{}.npy'.format(basename))
        duration = np.load(duration_path)

        sample = {
            'basename':basename,
            'speaker_id':speaker_id,
            'phone_id':phone_id,
            
            'mel':mel,
            'energy':energy,
            'f0':f0,
            'duration':duration
        }

        return sample

  
    def __len__(self):
        return len(self.basename)


    def process_meta(self, metafile):
        with open(os.path.join(self.feature_dir, metafile), 'r', encoding='utf-8') as f:
            basename = []
            speaker = []
            text = []
            phone = []
            
            for line in f.readlines():
                b, s, t, p = line.strip('\n').split('|')
                basename.append(b)
                speaker.append(s)
                text.append(t)
                phone.append(p)
                
        return basename, speaker, text, phone


    def reprocess(self, data, idxs):
        basenames = [data[idx]['basename'] for idx in idxs]
        speaker_ids = [data[idx]["speaker_id"] for idx in idxs]
        phone_ids = [data[idx]['phone_id'] for idx in idxs]
        
        mels = [data[idx]['mel'] for idx in idxs]
        f0s = [data[idx]['f0'] for idx in idxs]
        energys = [data[idx]['energy'] for idx in idxs]
        durations = [data[idx]['duration'] for idx in idxs]

        phone_id_lens = np.array([phone_id.shape[0] for phone_id in phone_ids])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speaker_ids = np.array(speaker_ids)
        phone_ids = pad_1D(phone_ids)
        mels = pad_2D(mels)
        energys = pad_1D(energys)
        f0s = pad_1D(f0s)
        durations = pad_1D(durations)

        return (
            basenames,
            speaker_ids,
            phone_ids,

            phone_id_lens,
            max(phone_id_lens),
            mel_lens,
            max(mel_lens),

            mels,
            energys,
            f0s,
            durations,
        )


    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d['phone_id'].shape[0] for d in data]) # [len(phone_id)]
            idx_arr = np.argsort(-len_arr) # [5(idx of long phone_id), 0, 1, 9, 11(short)]
        else:
            idx_arr = np.arange(data_size) # [0, 1, 2, ..., data_size-1]

        tail = idx_arr[data_size - (data_size % self.batch_size) :]
        body = idx_arr[: data_size - (data_size % self.batch_size)]
        body = body.reshape((-1, self.batch_size)).tolist() # body = [[,,],[,,],[,,]]
        if not self.drop_last and len(tail) > 0:
            body += [tail.tolist()] # body = [[,,],[,,],[,,],[,]]

        output = list()
        for idx in body:
            output.append(self.reprocess(data, idx)) # idx = [,,]

        return output


class TextDataset(Dataset):
    def __init__(self, metafile, preprocess_cfg):
        self.feature_dir = preprocess_cfg['path']['feature_dir']
        self.language = preprocess_cfg['text']['language']
        self.cleaner = preprocess_cfg['text']['text_cleaner']
        self.basename, self.speaker, self.text = self.process_meta(metafile)
        with open(os.path.join(self.feature_dir, "speakers.json")) as f:
            self.speaker_map = json.load(f)
    
    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        text = self.text[idx]
        if self.language == "en":
            phone_id = np.array(process_english(text)[-1])
        else:
            phone_id = np.array(process_mandarin(text)[-1])

        sample = {
            'basename':basename,
            'speaker_id':speaker_id,
            'phone_id':phone_id,
        }

        return sample

  
    def __len__(self):
        return len(self.basename)


    def process_meta(self, metafile):
        with open(metafile, 'r', encoding='utf-8') as f:
            basename = []
            speaker = []
            text = []
            
            for line in f.readlines():
                b, s, t = line.strip('\n').split('|')
                basename.append(b)
                speaker.append(s)
                text.append(t)
                
        return basename, speaker, text


    def collate_fn(self, data):
        basenames = [d['basename'] for d in data]
        speaker_ids = [d["speaker_id"] for d in data]
        phone_ids = [d['phone_id'] for d in data]

        phone_id_lens = np.array([phone_id.shape[0] for phone_id in phone_ids])

        speaker_ids = np.array(speaker_ids)
        phone_ids = pad_1D(phone_ids)

        return (
            basenames,
            speaker_ids,
            phone_ids,

            phone_id_lens,
            max(phone_id_lens)
        )


if __name__ == '__main__':
    import yaml
    import torch
    from torch.utils.data import DataLoader

    from utils import to_device
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    preprocess_cfg = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_cfg = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    # train
    train_meta = "train.txt"
    train_dataset = MyDataset(
        train_meta, preprocess_cfg, train_cfg, sort=True, drop_last=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    # val
    val_meta = "val.txt"
    val_dataset = MyDataset(
        val_meta, preprocess_cfg, train_cfg, sort=False, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )

    # train
    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )
    # val
    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
