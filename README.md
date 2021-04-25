# my-fastspeech2

## Usage

#### LJSpeech

**prepare_data:**
```bash
python prepare_data.py config/LJSpeech/preprocess.yaml
```

**mfa:**
```bash
./mfa/montreal-forced-aligner/bin/mfa_align ./preprocessed_data/LJSpeech/data ./mytext/lexicon/librispeech-lexicon.txt english ./preprocessed_data/LJSpeech/textgrid
```

**preprocess:**
```bash
python preprocess.py config/LJSpeech/preprocess.yaml
```

**train:**
```bash
python train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

**evaluate:**
```bash
python evaluate.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

**synthesize:**
```bash
python synthesize.py --source test-en.txt --restore_step 100000 -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

### AISHELL3

**prepare_data:**
```bash
python prepare_data.py config/AISHELL3/preprocess.yaml
```

**mfa:**
```bash
./mfa/montreal-forced-aligner/bin/mfa_train_and_align ./preprocessed_data/AISHELL3/data ./mytext/lexicon/pinyin-lexicon-r.txt ./preprocessed_data/AISHELL3/textgrid
```

**preprocess:**
```bash
python preprocess.py config/AISHELL3/preprocess.yaml
```

**train:**
```bash
python train.py -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

**evaluate:**
```bash
python evaluate.py -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

**synthesize:**
```bash
python synthesize.py --source test.txt --restore_step 3000 -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

## Reference

https://arxiv.org/abs/2006.04558

https://github.com/ming024/FastSpeech2