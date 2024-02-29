import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ['CriterionKD_ours']

class CriterionKD_ours(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature, mask_type, gamma):
        super(CriterionKD_ours, self).__init__()
        self.temperature = temperature
        self.mask_type = mask_type
        self.gamma = gamma
        
    def mask_generation(self, pred, soft):
        B, C, h, w = soft.size()
        # p_s = F.log_softmax(pred / self.temperature, dim=1)
        p_t = F.softmax(soft, dim=1)
        
        mask = []
        if self.mask_type == 'Ent':
            for i in range(B):
                Ent = torch.log(p_t[i]) * p_t[i,]
                Ent = -Ent.sum(dim=0)    
                mask.append(Ent)
            mask = torch.stack(mask)
        # least confidence  Focal loss style
        elif self.mask_type == 'LS':
            value, _ = p_t.max(dim=1)
            mask = (1 - value) **self.gamma

        # margin sample
        elif self.mask_type == 'MS':
            value, _ = p_t.topk(2, dim=1)
            mask = value[:,0,:,:] - value[:,1,:,:]
            mask = 2.72 - torch.exp(mask)

        elif self.mask_type == 'NLL':

            mask = F.log_softmax(soft, dim=1)
            mask, _ = torch.max(mask, dim=1)
            mask = -mask
        return mask.view(-1)
  
    def forward(self, pred, soft):

        B, C, H, W = soft.size()
        
        mask = self.mask_generation(pred, soft)
        
        scale_pred = pred.permute(0,2,3,1).contiguous().view(-1,C)
        scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C)
        
        p_s = F.log_softmax(scale_pred / self.temperature, dim=1)  # B*H*W X C
        p_t = F.softmax(scale_soft / self.temperature, dim=1)
        
        loss = (F.kl_div(p_s, p_t, reduction='none').sum(dim=1) * mask).sum() / (B * H * W) * (self.temperature**2)
    
        
        return loss