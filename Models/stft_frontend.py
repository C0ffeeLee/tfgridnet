import torch
import torch.nn as nn

class STFTFrontend(nn.Module):
    def __init__(self, n_fft=128, hop_length=64, win_length=128, window_type="hann", use_builtin_complex=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.use_builtin_complex = use_builtin_complex

        if window_type == "hann":
            self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        elif window_type == "hamming":
            self.register_buffer("window", torch.hamming_window(win_length), persistent=False)
        else:
            self.window = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, N, 1)
        _, N, M = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(B*M, N)
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect"
        )  # (B*M, F, T)
        F_bins, T_frames = spec.size(1), spec.size(2)
        spec = spec.view(B, M, F_bins, T_frames).permute(0, 1, 3, 2).contiguous()
        # 返回 (B, M, T, F)
        return spec
