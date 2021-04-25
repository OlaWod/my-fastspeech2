import yaml
import argparse

from myprocessor import ljspeech, aishell3


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_data(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_data(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader = yaml.FullLoader)
    main(config)

    '''

    !python prepare_data.py config/LJSpeech/preprocess.yaml
    
    !./mfa/montreal-forced-aligner/bin/mfa_align
     ./preprocessed_data/LJSpeech/data
     ./mytext/lexicon/librispeech-lexicon.txt
     english
     ./preprocessed_data/LJSpeech/textgrid

    !./mfa/montreal-forced-aligner/bin/mfa_train_and_align
     ./preprocessed_data/AISHELL3/data
     ./mytext/lexicon/pinyin-lexicon-r.txt
     ./preprocessed_data/AISHELL3/textgrid
     
    '''
