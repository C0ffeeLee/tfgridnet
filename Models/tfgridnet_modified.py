import math
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from stft_frontend import STFTFrontend
from istft_back import ISTFTDecoder
from utils import get_activation_layer


class TFGridNet(nn.Module):
    """
    独立版 TFGridNet（参考 ESPnet 实现，去除框架依赖）
    - 输入: mix_wav ∈ R^{B×N} 或 R^{B×N×M}
    - 输出: est_sources: List[Tensor]，每个元素 ∈ R^{B×L}，长度等于 n_srcs
    """
    def __init__(
        self,
        n_srcs: int = 2,
        n_fft: int = 128,
        stride: int = 64,
        window: str = "hann",
        n_imics: int = 1,
        n_layers: int = 6,
        lstm_hidden_units: int = 192,
        attn_n_head: int = 4,
        attn_approx_qk_dim: int = 512,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        activation: str = "prelu",
        eps: float = 1e-5,
        use_builtin_complex: bool = False,
        ref_channel: int = -1,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.ref_channel = ref_channel

        # —— STFT 前端 & ISTFT 后端 —— #
        self.stft = STFTFrontend(
            n_fft=n_fft, hop_length=stride, win_length=n_fft, window_type=window, use_builtin_complex=use_builtin_complex
        )
        self.istft = ISTFTDecoder(
            n_fft=n_fft, hop_length=stride, win_length=n_fft, window_type=window
        )

        # —— 2*M → emb_dim 的卷积层 —— #
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, kernel_size=ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        # —— 一系列 GridNetBlock —— #
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim=emb_dim,
                    emb_ks=emb_ks,
                    emb_hs=emb_hs,
                    n_freqs=n_freqs,
                    hidden_channels=lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
            )

        # —— 最后 Mask 估计的反卷积 —— #
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, kernel_size=ks, padding=padding)

    def forward(self, mix_wav: torch.Tensor) -> List[torch.Tensor]:
        """
        mix_wav: 
          - shape (B, N) 视为单通道
          - shape (B, N, M) 视为多通道（M 个麦克风）
        返回:
          est_list: List 长度 = n_srcs，每个元素 ∈ R^{B×L}
        """
        B = mix_wav.size(0)
        # 如果单通道，增加通道维度 M=1
        if mix_wav.dim() == 2:
            mix_wav = mix_wav.unsqueeze(-1)  # (B, N, 1)
        # RMS 归一化
        mix_std = torch.std(mix_wav, dim=(1, 2), keepdim=True)  # (B, 1, 1)
        x = mix_wav / (mix_std + 1e-8)  # (B, N, M)

        # → STFT，得到 complex 频谱 (B, M, T, F)
        spec = self.stft(x)  # (B, M, T, F), complex tensor

        # 拆成实部+虚部，拼成一个实数 Tensor (B, 2M, T, F)
        spec_real = spec.real  # (B, M, T, F)
        spec_imag = spec.imag  # (B, M, T, F)
        batch = torch.cat([spec_real, spec_imag], dim=1)  # (B, 2*M, T, F)

        # → 卷积 + 一系列 GridNetBlock → deconv
        batch = self.conv(batch)  # (B, emb_dim, T, F)
        for blk in self.blocks:
            batch = blk(batch)    # (B, emb_dim, T, F)
        batch = self.deconv(batch)  # (B, n_srcs*2, T, F)

        # reshape 得到 real/imag 部分：(B, n_srcs, 2, T, F)
        B_, _, T, F = batch.shape
        batch = batch.view(B, self.n_srcs, 2, T, F)
        real = batch[:, :, 0]  # (B, n_srcs, T, F)
        imag = batch[:, :, 1]  # (B, n_srcs, T, F)

        # 构造 complex 频谱 (B, n_srcs, T, F) complex
        real_flat = real.contiguous().view(B * self.n_srcs, T, F)
        imag_flat = imag.contiguous().view(B * self.n_srcs, T, F)
        spec_flat = torch.complex(real_flat, imag_flat)  # (B*n_srcs, T, F)
        spec_sep = spec_flat.view(B, self.n_srcs, T, F)  # (B, n_srcs, T, F)

        # ISTFT，把每个说话人的频谱 → 时域 (B, n_srcs, L)
        est_wavs = self.istft(spec_sep, target_len=x.size(1))  # (B, n_srcs, L)

        # 恢复 RMS
        est_wavs = est_wavs * mix_std  # (B, n_srcs, L)

        # 返回一个列表，每个元素 (B, L)
        est_list = [est_wavs[:, i, :] for i in range(self.n_srcs)]
        return est_list


    # @property
    # def num_spk(self):
    #     return self.n_srcs

    # @staticmethod
    # def pad2(input_tensor, target_len):
    #     input_tensor = torch.nn.functional.pad(
    #         input_tensor, (0, target_len - input_tensor.shape[-1])
    #     )
    #     return input_tensor


class GridNetBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        n_freqs: int,
        hidden_channels: int,
        n_head: int = 4,
        approx_qk_dim: int = 512,
        activation: str = "prelu",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

        in_channels = emb_dim * emb_ks

        # —— “帧内”RNN 分支 —— #
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            input_size=in_channels, hidden_size=hidden_channels,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            in_channels=hidden_channels * 2,
            out_channels=emb_dim,
            kernel_size=emb_ks,
            stride=emb_hs,
        )

        # —— “帧间”RNN 分支 —— #
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            input_size=in_channels, hidden_size=hidden_channels,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
            in_channels=hidden_channels * 2,
            out_channels=emb_dim,
            kernel_size=emb_ks,
            stride=emb_hs,
        )

        # —— 注意力分支 —— #
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)
        assert emb_dim % n_head == 0

        # 为每个 head 分别创建 Q、K、V 分支
        for ii in range(n_head):
            setattr(
                self,
                f"attn_conv_Q_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, kernel_size=1),
                    get_activation_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            setattr(
                self,
                f"attn_conv_K_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, kernel_size=1),
                    get_activation_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            setattr(
                self,
                f"attn_conv_V_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, kernel_size=1),
                    get_activation_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )

        # 最后把所有 head 拼回 emb_dim
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1),
            get_activation_layer(activation)(),
            LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=emb_dim, T, Q)  Q 即频率维度 F
        返回: out 同形状 (B, emb_dim, T, Q)
        """
        B, C, old_T, old_Q = x.shape
        # 计算 pad 后的 T、Q（保证能被 emb_hs 整除）
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x_padded = F.pad(x, (0, Q - old_Q, 0, T - old_T))  # pad dims: (left, right, top, bottom)

        ### —— 帧内（Frequency 方向）的 RNN 分支 —— ###
        intra_input = x_padded  # (B, C, T, Q)
        intra_normed = self.intra_norm(intra_input)  # (B, C, T, Q)
        # 把 (B, C, T, Q) → (B*T, C, Q)
        intra_rnn_in = intra_normed.transpose(1, 2).contiguous().view(B * T, C, Q)
        # Unfold：kernel=(emb_ks, 1)，stride=(emb_hs, 1) → (B*T, C*emb_ks, L1)
        intra_rnn_in = F.unfold(
            intra_rnn_in.unsqueeze(-1), (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # (B*T, C*emb_ks, L1)
        intra_rnn_in = intra_rnn_in.transpose(1, 2)  # (B*T, L1, C*emb_ks)
        intra_rnn_out, _ = self.intra_rnn(intra_rnn_in)  # (B*T, L1, 2*hidden)
        intra_rnn_out = intra_rnn_out.transpose(1, 2)  # (B*T, 2*hidden, L1)
        intra_rnn_out = self.intra_linear(intra_rnn_out)  # (B*T, emb_dim, Q)
        # reshape 回 (B, T, emb_dim, Q)
        intra_rnn_out = intra_rnn_out.view(B, T, C, Q).transpose(1, 2).contiguous()  # (B, C, T, Q)
        intra_rnn_out = intra_rnn_out + intra_input  # 残差连接

        ### —— 帧间（Time 方向）的 RNN 分支 —— ###
        inter_input = intra_rnn_out  # (B, C, T, Q)
        inter_normed = self.inter_norm(inter_input)  # (B, C, T, Q)
        # (B, C, T, Q) → (B, Q, C, T) → reshape为 (B*Q, C, T)
        inter_rnn_in = inter_normed.permute(0, 3, 1, 2).contiguous().view(B * old_Q, C, old_T)
        inter_rnn_in = F.unfold(
            inter_rnn_in.unsqueeze(-1), (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # (B*Q, C*emb_ks, L2)
        inter_rnn_in = inter_rnn_in.transpose(1, 2)  # (B*Q, L2, C*emb_ks)
        inter_rnn_out, _ = self.inter_rnn(inter_rnn_in)  # (B*Q, L2, 2*hidden)
        inter_rnn_out = inter_rnn_out.transpose(1, 2)  # (B*Q, 2*hidden, L2)
        inter_rnn_out = self.inter_linear(inter_rnn_out)  # (B*Q, emb_dim, T)
        inter_rnn_out = inter_rnn_out.view(B, old_Q, C, old_T).permute(0, 2, 3, 1).contiguous()  # (B, C, T, Q)
        inter_rnn_out = inter_rnn_out + inter_input  # 残差

        ### —— 注意力分支 —— ###
        attn_input = inter_rnn_out[:, :, :old_T, :old_Q]  # 回到 (B, C, old_T, old_Q)
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            # 动态获取对应子 module
            Q_i = getattr(self, f"attn_conv_Q_{ii}")(attn_input)  # (B, E, T, Q)
            K_i = getattr(self, f"attn_conv_K_{ii}")(attn_input)  # (B, E, T, Q)
            V_i = getattr(self, f"attn_conv_V_{ii}")(attn_input)  # (B, C//n_head, T, Q)
            all_Q.append(Q_i)
            all_K.append(K_i)
            all_V.append(V_i)
        # 拼接
        Q = torch.cat(all_Q, dim=0)  # ([n_head*B], E, T, Q)
        K = torch.cat(all_K, dim=0)  # ([n_head*B], E, T, Q)
        V = torch.cat(all_V, dim=0)  # ([n_head*B], C//n_head, T, Q)

        # Flatten time-frequency → time上做注意力
        B2, _, TT, QQ = V.shape  # B2 = n_head*B
        # Q, K: (B2, E, T, Q) → (B2, T, E*Q)
        Q2 = Q.transpose(1, 2).flatten(start_dim=2)  # (B2, T, E*Q)
        K2 = K.transpose(1, 2).flatten(start_dim=2)  # (B2, T, E*Q)
        # V: (B2, C//n_head, T, Q) → (B2, T, (C//n_head)*Q)
        V2 = V.transpose(1, 2).flatten(start_dim=2)  # (B2, T, (C//n_head)*Q)

        # 计算 scaled dot-product attention
        emb_dim_qk = Q2.shape[-1]
        attn_mat = torch.matmul(Q2, K2.transpose(1, 2)) / math.sqrt(emb_dim_qk)  # (B2, T, T)
        attn_mat = F.softmax(attn_mat, dim=-1)  # (B2, T, T)
        V3 = torch.matmul(attn_mat, V2)  # (B2, T, (C//n_head)*Q)

        # 恢复到 (B2, C//n_head, T, Q)
        V3 = V3.view(B2, (C // self.n_head), TT, QQ)  # (B2, C//n_head, T, Q)
        # 把各个 head concat 回去: (n_head*B, C//n_head, T, Q) → (B, n_head*(C//n_head), T, Q) = (B, C, T, Q)
        V3 = V3.view(self.n_head, B, (C // self.n_head), TT, QQ).transpose(0, 1).contiguous()
        V3 = V3.view(B, C, TT, QQ)  # (B, C, T, Q)
        V4 = self.attn_concat_proj(V3)  # (B, C, T, Q)

        out = V4 + inter_rnn_out[:, :, :old_T, :old_Q]  # 最后残差
        return out



class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension: int, eps: float = 1e-5):
        super().__init__()
        # gamma, beta 形状为 [1, input_dimension, 1, 1]
        param_size = [1, input_dimension, 1, 1]
        self.gamma = nn.Parameter(torch.ones(*param_size, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(*param_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, F)
        先在 C 维外对 T、F 两个维度求均值 var
        """
        if x.ndim != 4:
            raise ValueError(f"Expect x to have 4 dims, got {x.ndim}")
        B, C, T, F = x.shape
        # 在 dim=(1,) 上求 mean/var → 形状 (B,1,T,F)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.eps)
        x_hat = (x - mu) / std * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension: Tuple[int, int], eps: float = 1e-5):
        """
        input_dimension: (C', F) 
        gamma, beta 形状为 [1, C', 1, F]
        """
        super().__init__()
        assert len(input_dimension) == 2
        Cprime, F = input_dimension
        param_size = [1, Cprime, 1, F]
        self.gamma = nn.Parameter(torch.ones(*param_size, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(*param_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C', T, F)
        if x.ndim != 4:
            raise ValueError(f"Expect x to have 4 dims, got {x.ndim}")
        # 在 dim=(1, 3) 上求 mean/var → 形状 (B,1,T,1)
        mu = x.mean(dim=(1, 3), keepdim=True)
        var = x.var(dim=(1, 3), unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.eps)
        x_hat = (x - mu) / std * self.gamma + self.beta
        return x_hat

