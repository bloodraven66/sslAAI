import os
import torch
import librosa
import numpy as np
import scipy.stats
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from common.logger import logger
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.fastspeech import mask_from_lens
from common.wandb_logger import WandbLogger

os.environ["WANDB_SILENT"] = "true"
import common


class Operate():
    def __init__(self, params):
        self.patience = int(params.earlystopper.patience)
        self.counter = 0
        self.bestScore = None
        self.earlyStop = False
        self.valMinLoss = np.Inf
        self.delta = float(params.earlystopper.delta)
        self.minRun = int(params.earlystopper.minRun)
        self.numEpochs = int(params.common.numEpochs)
        self.modelName = params.common.model
        self.config = params
        if params.logging.disable:
            logger.info('wandb logging disabled')
            os.environ['WANDB_MODE'] = 'offline'
        self.logger = WandbLogger(params)
        self.expmode = params.common.expmode
        logger.info(f'Predicting {self.expmode}')
        self.best_cc = [-1, 0]
        self.break_mode = params.common.break_mode
        if self.break_mode:
            self.numEpochs = 1
            logger.info('Starting in break mode!')
        self.use_feats = params.common.use_feats
        
    def esCheck(self):
        score = -self.trackValLoss
        
        if self.epoch>self.minRun:
            if self.bestScore is None:
                self.bestScore = score
                self.saveCheckpoint()
            elif score < self.bestScore + self.delta:
                self.counter += 1
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.earlyStop = True
            else:
                self.bestScore = score
                self.saveCheckpoint()
                self.counter = 0

    def loadCheckpoint(self, chk=None):
        
        if chk is None: 
            
            if not self.config.common.finetune:
                raise NotImplementedError
            else:
                feat = self.config.common.feats if self.config.common.use_feats else 'baseline'
                chk = os.path.join('saved_models', 'pooled_10', feat+'_'+self.config.common.model_size+'_.pth')
                self.ft_chk = chk
        checkpoint = torch.load(chk)
        self.model.load_state_dict(checkpoint)
        logger.info(f'checkpoint loaded - {chk}')
        
    def saveCheckpoint(self, dry_run=False):
        if self.config.baseline:
            if self.config.common.use_feats:
                chk = self.config.common.feats
            else:
                chk = 'baseline'
            chk += f'_{self.config.common.model_size}_'
        else:
            
            chk = '_'.join([f'loss_{self.config.loss_type}',
                            f'loss_loc_{self.config.loss_loc}',
                            f'bs_{self.config.common.batch_size}',
                            f'phon_weight{self.config.phon_weight}',
                            f'{self.config.earlystopper.checkpoint_tag}'])
        
        if self.config.data.subjects == '1':
            if self.config.common.finetune:
                chk = 'ft_<' + Path(self.ft_chk).stem +'>_' +chk
            folder = os.path.join('saved_models', self.config.common.sub)
            chk = os.path.join(folder, f'{chk}.pth')
            if not os.path.exists(folder):
                os.mkdir(folder)
        elif self.config.data.subjects == '10':
            if self.config.common.finetune: raise NotImplementedError
            if not self.config.common.pooled: raise NotImplementedError
            if self.config.common.sub_embed: chk += '_subembed'
            folder = os.path.join('saved_models', 'pooled_10')
            chk = os.path.join(folder, f'{chk}.pth')
            if not os.path.exists(folder):
                os.mkdir(folder)
        else:
            raise NotImplementedError
        
        if not dry_run:
            logger.info(f'saving chk {chk}')
            torch.save(self.model.state_dict(), chk)


        
            self.valMinLoss = self.trackValLoss
        else:
            logger.info(f'chk {chk}')
    
    def trainloop(self, loader, mode, break_run=False, ):
        if self.break_mode: break_run = True
        if mode == 'train': self.model.train()
        elif mode == 'val': self.model.eval()
        else: raise NotImplementedError
        self.reset_iter_metrics()
        losses_to_upload = {'ema':[], 'dur':[], 'total':[]}
        with tqdm(loader, unit="batch", mininterval=20) as tepoch:
            for counter, data in enumerate(tepoch):
                if self.use_feats:
                    ema_padded, mfcc_padded, mel_padded, phon_padded, dur_padded, ema_lens, mel_lens, phon_lens, spkids, spks, names, phs, tphn_padded, stats, feat_padded = data
                else:
                    ema_padded, mfcc_padded, mel_padded, phon_padded, dur_padded, ema_lens, mel_lens, phon_lens, spkids, spks, names, phs, tphn_padded, stats = data
                if self.use_feats:
                    ema_padded, mfcc_padded, ema_lens, tphn_padded, phon_padded, phon_lens, spkids, feat_padded = self.set_device([ema_padded, mfcc_padded, ema_lens, tphn_padded, phon_padded, phon_lens, spkids, feat_padded], ignoreList=[])
                else:
                    ema_padded, mfcc_padded, ema_lens, tphn_padded, phon_padded, phon_lens, spkids = self.set_device([ema_padded, mfcc_padded, ema_lens, tphn_padded, phon_padded, phon_lens, spkids], ignoreList=[])
                if not self.config.common.sub_embed:  sp_id = None
                else: sp_id = spkids.long()
                if self.use_feats:
                    inputs = (feat_padded, ema_lens, spkids)
                else:
                    inputs = (mfcc_padded, ema_lens, spkids)
                model_out = self.model(inputs)
                feat_out, out_lens, dec_mask, phon_out, phon_out2, feat_out2 = model_out
                assert len(feat_out.shape) == 3
                targets = (ema_padded, ema_lens, tphn_padded, phon_padded, phon_lens)
                loss_dict = self.model.aai_loss(targets, model_out)
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss_dict["total"].backward()
                    self.optimizer.step()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                else:
                    metrics = self.get_cc(ema_padded.detach().cpu(), feat_out.detach().cpu(), ema_lens.detach().cpu().numpy().astype(int).tolist())
                    self.cc.extend(metrics[0])
                    self.rmse.extend(metrics[1])
                self.handle_metrics(loss_dict, mode)
                tepoch.set_postfix(loss=loss_dict["total"].item())
                if break_run:
                    break   
        
        # self.dump_cc()  #for stat test
        self.end_of_epoch(ema_padded, feat_out, ema_lens, mode)
        logger.info(f'cc:{self.cc}')
        return
    
    def dump_cc(self):
        folder = 'mfcc'
        if self.config.common.use_feats:
            folder = self.config.common.feats
        
        if self.config.data.subjects == '10':
            subfolder = 'pooled'
        elif self.config.data.subjects == '1':
            if self.config.common.finetune:
                subfolder = 'ft'
            else:
                subfolder = 'ss'
            subfolder = os.path.join(subfolder, self.config.common.sub)

        
        path = os.path.join(self.config.common.dump_cc, folder, subfolder) 
        if not os.path.exists(path) : os.makedirs(path)
        with open(os.path.join(path, f'{self.config.common.model_size}.npy'), 'wb') as f:
            np.save(f, self.cc)
        
        print(path)
        
    def end_of_epoch(self, ema, out, melLengths, mode):
        path = 'temp_files/file.txt'
        if mode == "val":
            self.logger.plot_ema(ema, out, melLengths)
            self.trackValLoss = sum(self.trackValLoss)/len(self.trackValLoss)
        for key in self.epoch_loss_dict:
            self.epoch_loss_dict[key] = sum(self.epoch_loss_dict[key])/len(self.epoch_loss_dict[key])
        if mode == "val":
            print(np.array(self.cc).shape)
            std = round(np.std(np.mean(self.cc, axis=0)), 3)
            da = np.mean(self.cc, axis=0).round(4).tolist()
            with open(path, 'a') as f:
                f.write(f'{"_".join([str(d) for d in da])}\n')
            self.cc = round(np.mean(np.mean(self.cc, axis=0)), 4)
            self.rmse = round(np.mean(np.mean(self.rmse, axis=0)), 4)
            if self.cc > self.best_cc[0]:
                self.best_cc = [self.cc, self.epoch]

            self.epoch_loss_dict['cc'] = self.cc
            print(f'{self.cc}({std})')

            
        self.logger.log(self.epoch_loss_dict)
        
    def reset_iter_metrics(self):
        self.epoch_loss_dict = {}
        self.skipped = 0
        self.trackValLoss = []
        self.cc, self.rmse = [], []


    def handle_metrics(self, iter_loss_dict, mode):
        for key in iter_loss_dict:
            if f'{key}_{mode}' not in self.epoch_loss_dict:
                self.epoch_loss_dict[f'{key}_{mode}'] = [iter_loss_dict[key].item()]
            else:
                self.epoch_loss_dict[f'{key}_{mode}'].append(iter_loss_dict[key].item())
        if mode == "val":
            self.trackValLoss.append(iter_loss_dict["aai"].item())


    def trainer(self, model, loaders):
        trainLoader, valLoader, testLoader = loaders
        self.optimizer, self.scheduler = self.get_trainers(model)
        self.model = model
        total_params = sum(
	        param.numel() for param in model.parameters()
        )
        logger.info(f'param count :{total_params}')



        if not self.config.common.infer:
            if self.config.common.finetune:
                
                self.loadCheckpoint(chk=self.config.common.finetune_chk)
            self.saveCheckpoint(dry_run=True)
            for epoch in range(int(self.numEpochs)):
                self.epoch = epoch
                self.trainloop(trainLoader, 'train')
                self.trainloop(valLoader, 'val')
                
                self.scheduler.step(self.trackValLoss)

                if self.expmode == "aai":
                    logger.info(f'[cc: {self.cc}]')

                self.esCheck()
                if self.earlyStop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            self.logger.summary({'best cc':self.best_cc[0], 'best cc epoch':self.best_cc[1]})
            
            logger.info('Training completed')

        else:
            self.epoch = 1
            logger.info('Starting Inference')
            feat = self.config.common.feats if self.config.common.use_feats else 'baseline'
            if self.config.data.subjects == '10':
                if self.config.common.pooled:
                    chk = os.path.join('saved_models', 'pooled_10', feat + '_' + self.config.common.model_size+'_.pth')
                    if not os.path.exists(chk):
                        chk = os.path.join('saved_models', 'pooled_10', feat +'.pth')
            elif self.config.data.subjects == '1':
                if self.config.common.finetune:
                    chk = os.path.join('saved_models', self.config.common.sub, 'ft_<'+feat + '_' + self.config.common.model_size+'_>_' + feat + '_' + self.config.common.model_size+'_.pth')
                    if not os.path.exists(chk):
                        chk = os.path.join('saved_models', self.config.common.sub, 'ft_<'+feat +'>_' + feat + '_' + self.config.common.model_size+'_.pth')
                else:
                    
                    chk = os.path.join('saved_models', self.config.common.sub, feat + '_' + self.config.common.model_size+'_.pth')
                    if not os.path.exists(chk):
                        chk = os.path.join('saved_models', self.config.common.sub, feat +'.pth')
            savename = feat+ self.config.common.model_size
            print(savename)
            self.model.load_state_dict(torch.load(chk, map_location='cpu'))
            self.trainloop(testLoader, 'val', break_run=False)
            self.logger.summary({'test cc':self.best_cc[0]})
            
            with open(os.path.join('temp_files', savename), 'a') as f:
                f.write(f'{self.best_cc[0]}\n')


    def get_cc(self, ema_, pred_, test_lens):
        ema_ = ema_.permute(0, 2, 1).numpy()
        pred_ = pred_.permute(0, 2, 1).numpy()
        m = []
        rMSE = []
        for j in range(len(pred_)):
            c  = []
            rmse = []
            for k in range(12):
                c.append(scipy.stats.pearsonr(ema_[j][k][:test_lens[j]], pred_[j][k][:test_lens[j]])[0])
                rmse.append(np.sqrt(np.mean(np.square(np.asarray(pred_[j][k][:test_lens[j]])-np.asarray(ema_[j][k][:test_lens[j]])))))
            m.append(c)
            rMSE.append(rmse)
        return m, rmse
    
    def get_trainers(self, model):
        if self.config.optimizer.name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.optimizer.lr), weight_decay=float(self.config.optimizer.weightdecay))
        else:
            raise Exception('Optimizer not found')
        

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, factor=0.6)
      
        return optimizer, scheduler

    def stats(data):
        print(torch.max(data), torch.min(data), torch.mean(data))

    def set_device(self, data, ignoreList):

        if isinstance(data, list):
            return [data[i].to(self.config.common.device).float() if i not in ignoreList else data[i] for i in range(len(data))]
        else:
            raise Exception('set device for input not defined')

    


    
    

            