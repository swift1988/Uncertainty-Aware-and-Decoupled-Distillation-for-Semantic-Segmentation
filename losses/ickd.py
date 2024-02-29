import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math

class ICKD(nn.Module):
    def __init__(self, s_channels, t_channels):
        super(ICKD, self).__init__()
        self.bin_h = 8
        self.bin_w = 8
        self.t_channels = t_channels
        self.s_channels = s_channels
        
        self.conv = nn.Conv2d(self.s_channels, self.t_channels, kernel_size=1, bias=False)


    def patch_split(self, input):
        B, C, H, W = input.size()
        bin_num_h = self.bin_h
        bin_num_w = self.bin_w
        
        if H == 45:
            bin_num_h = 9
            bin_num_w = 9
                    
        rH = H // bin_num_h
        rW = W // bin_num_w
        
        out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
        out = out.permute(0,2,4,3,5,1).contiguous()
        out = out.view(B, -1, C, rH, rW)
        
        out = out.view(-1, C, rH, rW) # [B * bin_num_h * bin_num_w, C, rH, rW]
        return out 
            
    def forward(self, s_feats, t_feats):
        if self.t_channels != self.s_channels:
            s_feats = self.conv(s_feats)
        
        s_feats = self.patch_split(s_feats)
        t_feats = self.patch_split(t_feats)
        
        b, c = s_feats.shape[0], s_feats.shape[1]
        
        f_s = s_feats.view(b, c, -1)
        f_t = t_feats.view(b, c, -1)
        
        emd_s = torch.bmm(f_s, f_s.permute(0,2,1))
        emd_s = torch.nn.functional.normalize(emd_s, dim = 2)

        emd_t = torch.bmm(f_t, f_t.permute(0,2,1)) 
        emd_t = torch.nn.functional.normalize(emd_t, dim = 2)
        
        G_diff = emd_s - emd_t
        loss = (G_diff * G_diff).view(b, -1).sum() / (c * b)
        return loss