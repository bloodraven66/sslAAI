import os
from common.utils import get_files
from common.logger import logger
import scipy.io
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from multiprocessing import Pool
from tqdm import tqdm
from librosa.filters import mel as librosa_mel_fn

class Dataset_prepare_dump(torch.utils.data.Dataset):
    def __init__(self, mode, common, **kwargs):
        self.__dict__.update(kwargs)
        if mode == 'train':
            self.prepare_data()
        if self.subjects == '10':
            self.subs = [] #10 selected subject names
        elif self.subjects == 'all':
            self.subs = os.listdir(os.path.join(self.rootPath, 'DataBase'))
        elif self.subjects == '1':
            self.subs = [common.sub]
        else:
            raise NotImplementedError
        self.parse_file_based_on_mode(mode)
        self.return_keys = [
                            'filename',
                            'ema',
                            'ema_lengths',
                            'mfcc_norm',
                            'mfcc_length',
                            'mel_norm',
                            'mel_length',
                            'phons',
                            'phon_ids',
                            'dur',
                            'tphn_ids',
                            'beginend',
                            'spkid',
                            'speaker'
                            ]
        if common.use_feats:
            self.parse_feast_paths(common)
        else:
            self.feat_paths = None
    
    def parse_feast_paths(self, common):
        feat = common.feats
        folder = common.dump_loc
        self.feat_paths = {}
        folder = os.path.join(folder, feat)
        for spk_idx, sub in enumerate(sorted(self.subs)):
            f = os.path.join(folder, sub)
            for l in os.listdir(f):
                self.feat_paths[Path(l).stem] = str(os.path.join(f, l))
                
    def parse_file_based_on_mode(self, mode):
        files = []
        for spk_idx, sub in enumerate(sorted(self.subs)):
            for idx, wavfile in enumerate(sorted(os.listdir(os.path.join(self.rootPath, 'DataBase', sub, self.wavFolder)))):
                if ((idx + 10) % 10)==0:
                    if mode == 'test': 
                        files.append(Path(wavfile).stem)
                elif ((idx+10-1)%10)==0:
                    if mode == 'val': 
                        files.append(Path(wavfile).stem)
                else:
                    if mode == 'train': 
                        files.append(Path(wavfile).stem)
        available_files = {Path(f).stem:f for f in get_files(self.dumpdir, '.pt')}
        self.files = [available_files[f] for f in files] 
        logger.info(f'{len(self.files)} data being used')
        
    def __len__(self):
        return len(self.files)     
            
    def __getitem__(self, idx):
        
        data = torch.load(self.files[idx])
        # print(data.keys(), self.return_keys)
        # for d in data:
        #     if d in self.return_keys:
        #         print(d)
        # print(data['speaker'], data['spkid'])
        data = [data[d] for d in data if d in self.return_keys]
        if self.feat_paths is not None:
            feat = torch.load(self.feat_paths[Path(self.files[idx]).stem])
            data = data + [feat[-1]]
        # print(len(data))
        return data
     
    def prepare_data(self):
        if not os.path.exists(self.dumpdir): os.mkdir(self.dumpdir)
        all_subs = os.listdir(os.path.join(self.rootPath, 'DataBase'))
        available_files = get_files(self.dumpdir, '.pt')
        logger.info(f'{len(available_files)} files found in dumpdir {self.dumpdir}')
        if len(available_files) == len(all_subs)*460:
            logger.info('All files generated, returning')
            return
        logger.info('Extracting feats')
        wav_paths = [str(f) for f in get_files(os.path.join(self.rootPath, 'DataBase'), '.wav')]
        F_all = {}
        begin_end_all = {}
        spk_id = {}
        for spk_idx, sub in enumerate(sorted(all_subs)):
            spk_id[sub] = spk_idx
            F_all[sub] = {}
            for idx, wavfile in enumerate(sorted(os.listdir(os.path.join(self.rootPath, 'DataBase', sub, self.wavFolder)))):
                F_all[sub][Path(wavfile).stem] = idx
            begin_end_path = os.path.join(self.startStopFolder, sub)
            begin_end = scipy.io.loadmat(os.path.join(begin_end_path, os.listdir(begin_end_path)[0]))
            begin_end_all[sub] = begin_end['BGEN']
        self.phon_dict = np.load(self.phonFile, allow_pickle=True)
        self.word_to_int = self.phon_dict['wti'].item()
        self.int_to_word = {}

        for key in self.word_to_int:
            self.word_to_int[key] += 1
            self.int_to_word[self.word_to_int[key]] = key

        def read_transcript(path, begin, end):
            df = pd.read_csv(path, header=None)
            info = list(df[0].map(lambda x:x.split()))
            durs = [(int(float(info[i][0])*100), int(float(info[i][1])*100), info[i][-1]) for i in range(len(info))]
            actual_durs = durs[:]
            total_dur = durs[-1][1] - durs[0][0]
            start = durs[0][1]
            stop = durs[-1][0]

            for ph_loop_idx in range(len(durs)):
                if begin <= durs[ph_loop_idx][1]:
                    break
            durs = durs[ph_loop_idx:]
            if durs[0][1] == begin:
                durs[0] = (0, begin, 'sil')
            else:
                durs[0] = (begin, durs[0][1], durs[0][-1])
                durs = [(0, begin, 'sil')] + durs

            start = durs[0][1]
            stop = durs[-1][0]

            if start != begin or durs[0][-1] != 'sil':
                raise Exception('debug!')

            if stop != end or durs[-1][-1] != 'sil':
                for ph_loop_idx in range(len(durs)):
                    if end <= durs[ph_loop_idx][1]:
                        break
                durs = durs[:ph_loop_idx+1]
                durs[-1] = (durs[-1][0], end, durs[-1][-1])
                durs = durs + [(end, end+10, 'sil')]
                stop = durs[-1][0]

            phonemes, phoneme_ids, tphn, tphn_ids, durations = [], [], [], [], []
            for (ph_start, ph_end, ph) in durs[1:-1]:
                phonemes.append(ph)
                phoneme_ids.append(self.word_to_int[ph])
                durations.append(ph_end-ph_start)
                tphn.extend([ph]*(durations[-1]))
                tphn_ids.extend([self.word_to_int[ph]]*(durations[-1]))
            for current_dur in durations:
                    assert current_dur>=1
            return phonemes, phoneme_ids, durations, tphn, tphn_ids

        def extract_feats(filename):
            filestem = Path(filename).stem
            subject = filename.split('/')[-4]
            speaker_id = spk_id[subject]
            F = F_all[subject][filestem]
            beginEnd = begin_end_all[subject]
            ema_mat = scipy.io.loadmat(os.path.join(self.rootPath, 'DataBase', subject, self.emaFolder, filestem))
            ema_temp = ema_mat['EmaData'];
            ema_temp = np.transpose(ema_temp)
            ema_temp2 = np.delete(ema_temp, [4,5,6,7,10,11],1)
            mean_of_data = np.mean(ema_temp2,axis=0)
            ema_temp2 -= mean_of_data
            C = 0.5*np.sqrt(np.mean(np.square(ema_temp2),axis=0))
            ema_norm = np.divide(ema_temp2,C)
            ema = ema_temp
            assert ema.shape[-1] == 18
            assert ema_norm.shape[-1] == 12
            [aE,bE] = ema.shape
            begin = np.int(beginEnd[0, F]*100)
            end = np.int(beginEnd[1, F]*100)
            mfcc = np.load(os.path.join(self.rootPath, self.mfccFolder, subject, filestem+'.npy')).T
            assert mfcc.shape[1] == 13
            mean_G = np.mean(mfcc, axis=0)
            std_G = np.std(mfcc, axis=0)
            mfcc_norm = self.stdFrac*(mfcc-mean_G)/std_G
            mfcc_norm = mfcc_norm[begin:end, :]
            mfcc = mfcc[begin:end, :]
            ema = ema[begin:end, :]
            ema_norm = ema_norm[begin:end, :]
            transcript_path = os.path.join(self.rootPath, 'FA_EN_ALL', subject, self.alignFolder)
            phonemes, phoneme_ids, durations, tphn, tphn_ids = read_transcript(os.path.join(transcript_path, sorted(os.listdir(transcript_path))[F]), begin, end)
            y, sr = librosa.load(filename, sr=self.sampleRate)
            y = y[int(sr*begin/100): int(sr*end/100)]
            y = torch.from_numpy(y)
            y = torch.clamp(y, min=-1, max=1).numpy()
            spec = np.abs(librosa.stft(y=y, n_fft=self.nfft, hop_length=self.hopLength, win_length=self.winLength))
            mel = librosa.feature.melspectrogram(S=spec, sr=self.sampleRate, n_fft=self.nfft, n_mels=self.nMels, fmin=self.fMin, fmax=self.fMax).T
            mel_norm = np.clip(mel, a_min=1.e-5, a_max=None)
            mel_norm = np.log(mel_norm)
            data = {
                    'filename':filename,
                    'ema':ema,
                    'ema_norm':ema_norm,
                    'ema_lengths':ema_norm.shape[0],
                    'mfcc':mfcc,
                    'mfcc_norm':mfcc_norm,
                    'mfcc_length':mfcc.shape[0],
                    'mel':mel,
                    'mel_norm':mel_norm,
                    'mel_length':mel_norm.shape[0],
                    'phons':phonemes,
                    'phon_ids':phoneme_ids,
                    'dur':durations,
                    'tphn':tphn,
                    'tphn_ids':tphn_ids,
                    'beginend':(begin,end),
                    'spkid':speaker_id,
                    'speaker':subject,
                    }
            if not os.path.exists(os.path.join(self.dumpdir, subject)):
                os.mkdir(os.path.join(self.dumpdir, subject))
            torch.save(data, os.path.join(self.dumpdir, subject, filestem+'.pt'))
            return
        for filename in tqdm(wav_paths):
            extract_feats(filename)
        
        
class Dataset_prepare_dump_collate():
    
    def __init__(self, feat=False):
        self.feat = feat

    def __call__(self, batch):
        #name, ema, lens, mfcc, mfcclen, mel, mellen, ph, phid, dur, spkid, spk
        ema_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[1]) for x in batch]),
            dim=0, descending=True)
        max_ema_len = ema_lengths[0]
        ema_padded = torch.FloatTensor(len(batch), max_ema_len, 12)
        ema_padded.zero_()
        mfcc_padded = torch.FloatTensor(len(batch), max_ema_len, 13)
        mfcc_padded.zero_()
        tphn_padded = torch.FloatTensor(len(batch), max_ema_len)
        tphn_padded.zero_()
        max_mel_len = max([x[5].shape[0] for x in batch])
        mel_padded = torch.FloatTensor(len(batch), max_mel_len, 80)
        mel_padded.zero_()
        max_phon_len = max([len(x[8]) for x in batch])
        phon_padded = torch.LongTensor(len(batch), max_phon_len)
        phon_padded.zero_()
        dur_padded = torch.LongTensor(len(batch), max_phon_len)
        dur_padded.zero_()
        ema_lens_, mel_lens_, phon_lens_, spkids, spks, names, phs = [], [] ,[], [], [], [], []
        stats = []
        if self.feat:
            dim = batch[ids_sorted_decreasing[0]][-1].shape[-1]
            feat_padded = torch.FloatTensor(len(batch), max_ema_len, dim)
            feat_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            name = batch[ids_sorted_decreasing[i]][0]
            names.append(name)
            ema = batch[ids_sorted_decreasing[i]][1] 
            ema_temp2 = np.delete(ema, [4,5,6,7,10,11],1)
            mean_of_data = np.mean(ema_temp2,axis=0)
            ema_temp2 -= mean_of_data
            C = 0.5*np.sqrt(np.mean(np.square(ema_temp2),axis=0))
            ema = np.divide(ema_temp2,C)
            stats.append((C, mean_of_data))
            # print(ema.shape)
            # 4,5,6,7,10,11
            
            ema_padded[i, :ema.shape[0], :] = torch.from_numpy(ema)
            ema_lens = batch[ids_sorted_decreasing[i]][2]
            ema_lens_.append(ema_lens)
            mfcc = batch[ids_sorted_decreasing[i]][3]
            mfcc_padded[i, :mfcc.shape[0], :] = torch.from_numpy(mfcc)
            mfcc_lens = batch[ids_sorted_decreasing[i]][4]
            mel = batch[ids_sorted_decreasing[i]][5]
            mel_padded[i, :mel.shape[0], :] = torch.from_numpy(mel)
            mel_len = batch[ids_sorted_decreasing[i]][6]
            mel_lens_.append(mel_len)
            ph = batch[ids_sorted_decreasing[i]][7]
            phs.append(ph)
            phid = t(batch[ids_sorted_decreasing[i]][8])
            phon_lens_.append(len(phid))
            phon_padded[i, :phid.shape[0]] = phid
            dur = t(batch[ids_sorted_decreasing[i]][9])
            dur_padded[i, :dur.shape[0]] = dur
            spkid = batch[ids_sorted_decreasing[i]][12]
            spkids.append(spkid)
            spk = batch[ids_sorted_decreasing[i]][13]
            spks.append(spk)
            tphn = batch[ids_sorted_decreasing[i]][10]
            tphn_padded[i, :len(tphn)] = t(tphn)
            
            if self.feat:
                feat = batch[ids_sorted_decreasing[i]][-1]
                start, stop = batch[ids_sorted_decreasing[i]][11]
                feat = feat[start:stop]
                feat_padded[i, :ema.shape[0], :] = feat
        stats = torch.from_numpy(np.array(stats))
        if self.feat:
            return ema_padded, mfcc_padded, mel_padded, phon_padded, dur_padded, t(ema_lens_), t(mel_lens_), t(phon_lens_), t(spkids), spks, names, phs, tphn_padded, stats, feat_padded
        return ema_padded, mfcc_padded, mel_padded, phon_padded, dur_padded, t(ema_lens_), t(mel_lens_), t(phon_lens_), t(spkids), spks, names, phs, tphn_padded, stats

def t(arr):
    if isinstance(arr, list):
        return torch.from_numpy(np.array(arr))
    elif isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    else:
        raise NotImplementedError