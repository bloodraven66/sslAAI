from attrdict import AttrDict
import yaml
import torch
import numpy as np
from trainer import ema_trainer
from models import fastspeec
import librosa
from pathlib import Path

def read_yaml(yamlFile):
    with open(yamlFile) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg

def t_(dataset):
    return torch.from_numpy(np.array(dataset))


def get_trainer(config):
    if config.common.expmode == 'aai':
        return ema_trainer
    else:
        raise NotImplementedError

def get_model(config):
    modelChoice = config.common.model
    mode = config.common.expmode
    subs = config.common.num_speakers
    if not config.common.use_feats:
        input_dim = 13
    else:
        if config.common.feats in ['pase_plus',]:
            input_dim = 256
        elif config.common.feats in ['audio_albert', 'tera', 'mockingjay']:
            input_dim = 768 
        elif config.common.feats in ['vq_wav2vec', 'wav2vec', 'apc', 'npc']:
            input_dim = 512
        elif config.common.feats in ['decoar']:
            input_dim = 2048
        else:
            raise NotImplementedError
    if modelChoice == 'fastspeech':
        
        
        if mode == 'aai':
            if config.common.model_size == 'default':
                model_config = read_yaml('config/fs.yaml')
            elif config.common.model_size == 'small':
                model_config = read_yaml('config/fs_small.yaml')
            elif config.common.model_size == 'large':
                model_config = read_yaml('config/fs_large.yaml')
            model = fastspeech.FastSpeech(n_mel_channels=12,
                                            input_dim=input_dim,
                                            baseline=config.baseline,
                                            use_spk_embed=config.common.sub_embed,
                                            padding_idx=config.data.phonPadValue,
                                            n_speakers=subs,
                                            loss_type=config.loss_type,
                                            loss_loc=config.loss_loc,
                                            phon_weight=config.phon_weight,
                                            **model_config).to(config.common.device)
        
        else:
            raise Exception(f'Training mode {mode} not defined')
   
    else:
        raise Exception('model Not found')


    return model


def load_pretrained(config, model):
    model.load_state_dict(torch.load(config.common.ema_pretrained))
    return model

MAX_WAV_VALUE = 32768.0

def get_audio(sample, lengths):
    y_gen_tst = sample[:int(lengths[0])].T
    y_gen_tst = np.exp(y_gen_tst)
    S = librosa.feature.inverse.mel_to_stft(
            y_gen_tst,
            power=1,
            sr=22050,
            n_fft=1024,
            fmin=0,
            fmax=8000.0)
    audio = librosa.core.griffinlim(
            S,
            n_iter=32,
            hop_length=256,
            win_length=1024)
    audio = audio * MAX_WAV_VALUE
    audio = audio.astype('int16')
    return audio

def find_audio(name, folder):
    all_files = set(get_files(folder))
    filename  = [str(f) for f in all_files if name in str(f)    ]
    assert len(filename) == 1
    return filename[0]

def get_files(path, extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

