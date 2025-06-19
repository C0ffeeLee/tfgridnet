# data.py
import os
import torch
import soundfile as sf

class WSJ0MixDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subset, n_fft=512, hop_length=128, win_length=512):
        """
        root_dir: 数据集根目录，如 'wsj0_2mix'
        subset: 'tr', 'cv' 或 'tt'
        """
        super().__init__()
        self.mix_dir = os.path.join(root_dir, subset, 'mix_clean')
        self.s1_dir  = os.path.join(root_dir, subset, 's1')
        self.s2_dir  = os.path.join(root_dir, subset, 's2')
        self.file_list = sorted(os.listdir(self.mix_dir))
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # 预创建窗函数
        self.window = torch.hann_window(win_length)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        fname = self.file_list[idx]
        # 读取混合语音，以及两个参考
        mix_path = os.path.join(self.mix_dir, fname)
        s1_path  = os.path.join(self.s1_dir , fname)
        s2_path  = os.path.join(self.s2_dir , fname)
        
        mix_wav, _ = sf.read(mix_path, dtype='float32')
        s1_wav , _ = sf.read(s1_path , dtype='float32')
        s2_wav , _ = sf.read(s2_path , dtype='float32')
        
        mix = torch.from_numpy(mix_wav).float()  # shape: (L,)
        s1  = torch.from_numpy(s1_wav ).float()
        s2  = torch.from_numpy(s2_wav ).float()
        
        # 计算 STFT (返回复数 tensor)
        X = torch.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length,
                       win_length=self.win_length, window=self.window,
                       return_complex=True)  # shape: (F, T)
        
        # 也可以预先把参考信号做 STFT，但常见做法是在时域计算 SI-SDR
        return {
            'mix_wav': mix,       # 用于时域 SI-SDR 评估
            's1_wav' : s1,
            's2_wav' : s2,
            'mix_spec': X         # 网络输入
        }
