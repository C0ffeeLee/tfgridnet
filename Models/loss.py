# loss.py
import torch
import torch.nn as nn

EPS = 1e-8

def si_sdr(mix, src, est_src, zero_mean=True):
    """
    mix:     shape (B, L), 混合语音（时域）
    src:     shape (B, C, L), 真实参考信号 (C=说话人数)
    est_src: shape (B, C, L), 估计的分离信号
    """
    B, C, L = est_src.shape
    if zero_mean:
        mean_src = src.mean(dim=-1, keepdim=True)
        mean_est = est_src.mean(dim=-1, keepdim=True)
        src_z = src - mean_src
        est_z = est_src - mean_est
    else:
        src_z = src
        est_z = est_src

    # 逐个说话人计算 SI-SDR
    s_target = (torch.sum(est_z * src_z, dim=-1, keepdim=True) * src_z) \
               / (torch.sum(src_z * src_z, dim=-1, keepdim=True) + EPS)
    e_noise  = est_z - s_target

    target_norm = torch.sum(s_target ** 2, dim=-1)  # (B, C)
    noise_norm  = torch.sum(e_noise  ** 2, dim=-1)  # (B, C)

    si_sdr_val = 10 * torch.log10((target_norm + EPS) / (noise_norm + EPS))  # (B, C)
    # 平均所有说话人
    return -si_sdr_val.mean()  # 负值作为要最小化的损失

class SISDRLoss(nn.Module):
    def __init__(self, zero_mean=True):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean

    def forward(self, mix, src, est_src):
        return si_sdr(mix, src, est_src, self.zero_mean)

