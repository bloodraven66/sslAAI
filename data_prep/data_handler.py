# from data_prep.loadData import Dataset
from data_prep.loadData_v2 import Dataset_prepare_dump, Dataset_prepare_dump_collate
from torch.utils.data import DataLoader
import s3prl.hub as hub
import torch
from pathlib import Path
import os
import librosa
from tqdm import tqdm

def collect(cfg):
    feat = cfg.common.use_feats
    collate_fn = Dataset_prepare_dump_collate(feat)
    if cfg.common.dump_feats:
        dump_feats(cfg)
    loaders = []
    for mode in ['train', 'val', 'test']:
        dataset = Dataset_prepare_dump(mode, cfg.common, **cfg.data )
        loader_ = DataLoader(   
                            dataset, 
                            shuffle=True if mode == 'train' else False, 
                            batch_size=int(cfg.common.batch_size), 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            )
        loaders.append(loader_)
    # global_stats(loaders[0])
    # exit()
    return loaders

# def global_stats(loader):
#     for data in loader:
#         ema_padded, mfcc_padded, mel_padded, phon_padded, dur_padded, ema_lens, mel_lens, phon_lens, spkids, spks, names, phs, tphn_padded = data

def get_files(path, extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

def dump_feats(cfg):
    feat = cfg.common.feats
    wav_folder = os.path.join(cfg.data.rootPath, 'DataBase')
    wavs = get_files(wav_folder)
    try:
        model = getattr(hub, feat)()
    except:
        raise Exception(f'verify {feat} for pretrained s3prl')
    
    device = cfg.common.device
    model = model.to(device)
    folder = os.path.join(cfg.common.dump_loc, feat)
    print(f'saving at {folder}')
    for p in tqdm(wavs):
        p = str(p)
        y, sr = librosa.load(p, sr=16000)
        y = torch.from_numpy(y).to(device)
        with torch.no_grad():
            reps = model([y])["hidden_states"]
        reps = torch.cat(reps, 0).cpu()
        spk = p.split('/')[-4]
        name = Path(p).stem
        save_path = os.path.join(folder, spk)
        if not os.path.exists(save_path): os.makedirs(save_path)
        save_path = os.path.join(save_path, name+'.pt')
        torch.save(reps, save_path)
    exit()