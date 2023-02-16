from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *
from common.logger import logger

class FastSpeech(nn.Module):
    def __init__(self, n_mel_channels, padding_idx,input_dim,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head, loss_type, loss_loc, phon_weight,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size, p_in_fft_dropout, p_in_fft_dropatt, 
                 p_in_fft_dropemb,out_fft_output_size, baseline,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 n_speakers, speaker_emb_weight, use_spk_embed):
        super(FastSpeech, self).__init__()
        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim,
            padding_idx=padding_idx)
        self.phon_weight = phon_weight
        n_speakers = int(n_speakers)
        if use_spk_embed:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
            logger.info('Using speaker embed')
        else:
            self.speaker_emb = None
            logger.info('No speaker embed')

        self.speaker_emb_weight = speaker_emb_weight
        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )  
        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)
        self.tag = "aai"
        self.relu = nn.ReLU()
        self.ema_loss_fn_sum = nn.MSELoss(reduction='none')
        self.mfcc_proj = nn.Linear(input_dim, symbols_embedding_dim)
        self.baseline = baseline
        
        
    def forward(self, inputs, use_gt_pitch=True, pace=1.0, max_duration=75):
        (inputs, out_lens, speaker) = inputs
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker.long()).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)
        inputs = self.mfcc_proj(inputs)
        enc_out, enc_mask = self.encoder(inputs, out_lens, conditioning=spk_emb)
        dec_out, dec_mask = self.decoder(enc_out, out_lens)
        feat_out = self.proj(dec_out)
        return feat_out, out_lens, dec_mask, None, None, None
    
    def aai_loss(self, inputs, targets):
        loss_dict = {}
        ema_padded, ema_lens, tphn, phon_padded, phon_lens  = inputs
        feat_out, ema_lens, mask, phon_out, phon_out2, feat_out2 = targets
        lengths_for_ema = []
        assert feat_out.shape[-1] == 12
        assert ema_padded.shape[-1] == 12
        
       
        mask = mask.float()
        loss = (self.ema_loss_fn_sum(feat_out, ema_padded) * mask) / mask.sum(1).unsqueeze(1)
        loss = torch.sum(torch.mean(loss, 0)) / feat_out.shape[-1]

      
        loss_dict[self.tag] = loss
        loss_dict["total"] = loss

