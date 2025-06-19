import torch
import torch.nn as nn

class ISTFTDecoder(nn.Module):
    def __init__(self, n_fft=128, hop_length=64, win_length=128, window_type="hann"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        if window_type == "hann":
            self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        elif window_type == "hamming":
            self.register_buffer("window", torch.hamming_window(win_length), persistent=False)
        else:
            self.window = None

    def forward(self, spec_complex: torch.Tensor, lengths: torch.Tensor=None, target_len: int=None) -> torch.Tensor:
        B, C, T, F = spec_complex.shape
        spec_flat = spec_complex.permute(0,1,3,2).contiguous().view(B*C, F, T)
        wav = torch.istft(
            spec_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=target_len,
            center=True
        )  # (B*C, L)
        L = wav.size(-1)
        wav = wav.view(B, C, L)
        return wav
