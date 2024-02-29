import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator


__all__ = ['CriterionMaskDecoupleStructuralKD']


class CriterionMaskDecoupleStructuralKD(nn.Module):
    def __init__(self, ignore_label, num_class):
        super(CriterionMaskDecoupleStructuralKD, self).__init__()
        self.ignore_label = ignore_label
        self.num_class = num_class

    def prior_map(self, targets):
        targets = targets.unsqueeze(dim=0) if len(targets.shape) == 2 else targets
        res = torch.stack([targets == c for c in range(self.num_class)], dim=1).double()  # B C H W
        
        b, c, h, w = res.shape
        
        intra = res.reshape(b, c, -1)  # B, C, N(HW)
        
        intra = torch.matmul(intra.permute(0, 2, 1), intra) # B, N, N 
        inter = 1 - intra  
        
        return intra, inter
    
    def pair_wise_sim_map(self, fea):
        B, C, H, W = fea.size()
        fea = fea.reshape(B, C, -1)
        fea_T = fea.transpose(1,2)

        # dot product
        sim_map = torch.bmm(fea_T, fea)
        # embed
        # sim_map = torch.exp(sim_map)
        return sim_map

    def forward(self, feat_S, feat_T, targets):
        
        B, C, H, W = feat_S.size()

        patch_h, patch_w = 2, 2
        
        maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        
        labels = targets.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (feat_S.shape[2], feat_S.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long() # B, H, W
        intra_mask, inter_mask = self.prior_map(labels)
        
        mask = torch.where(labels == self.ignore_label, 0.0, 1.0)  # filter ignore_label 
        mask = mask.view(B, -1).unsqueeze(-1)  # B, N (HW), 1 
        
        intra_mask, inter_mask = intra_mask * mask, inter_mask * mask
        
        p_intra, p_inter = intra_mask.sum(dim=(1,2)) + 1e-12, inter_mask.sum(dim=(1,2)) + 1e-12

        
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        
        S_sim_map = self.pair_wise_sim_map(feat_S)
        T_sim_map = self.pair_wise_sim_map(feat_T)
        
        B, H, W = S_sim_map.size()

        sim_err = ((S_sim_map - T_sim_map)**2)

        intra_sim = (sim_err * intra_mask).sum(dim=(1,2)) / p_intra
        inter_sim = (sim_err * inter_mask).sum(dim=(1,2)) / p_inter
        
        del intra_mask
        del inter_mask
        
        return intra_sim.mean(), inter_sim.mean()

if __name__ == "__main__":
    
    feat_s = torch.randn(1, 3, 3, 3)
    feat_t = torch.randn(1, 3, 3, 3)

    targets = torch.tensor([[1, 1, -1], [2, 0, -1], [8, 7, 9]]).unsqueeze(0)
    
    mask = (torch.rand(1, 3, 3) > 0.5).float()
    print(mask)  
    skd = CriterionMaskDecoupleStructuralKD(-1, 10)
    
    skd(feat_s, feat_t, mask, targets)

    
    
    
