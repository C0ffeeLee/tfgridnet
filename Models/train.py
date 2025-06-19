# train.py（仅示例核心片段）
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import WSJ0MixDataset  # 前面示例已给 data.py
from tfgridnet_modified import TFGridNet
from loss import SISDRLoss
from config import config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    train_set = WSJ0MixDataset(root_dir='wsj0_2mix', subset='tr',
                               n_fft=config['n_fft'], hop_length=config['hop_length'],
                               win_length=config['win_length'])
    cv_set    = WSJ0MixDataset(root_dir='wsj0_2mix', subset='cv',
                               n_fft=config['n_fft'], hop_length=config['hop_length'],
                               win_length=config['win_length'])
    train_loader = DataLoader(train_set, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    cv_loader    = DataLoader(cv_set,    batch_size=config['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    # 模型、损失、优化器
    model = TFGridNet(
        n_srcs=config['n_speakers'],
        n_fft=config['n_fft'],
        stride=config['hop_length'],
        window='hann',
        n_imics=1,
        n_layers=config['n_layers'],
        lstm_hidden_units=config['lstm_hidden_units'],
        attn_n_head=config['attn_n_head'],
        attn_approx_qk_dim=config['attn_approx_qk_dim'],
        emb_dim=config['emb_dim'],
        emb_ks=config['emb_ks'],
        emb_hs=config['emb_hs'],
        activation=config['activation'],
        eps=1e-5,
        use_builtin_complex=False,
    )
    model.to(device)

    criterion = SISDRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    os.makedirs(config['ckpt_dir'], exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, config['epochs'] + 1):
        # —— 训练一轮 —— #
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            mix_wav = batch['mix_wav'].to(device)  # (B, L)
            s1_wav  = batch['s1_wav'].to(device)
            s2_wav  = batch['s2_wav'].to(device)

            # TFGridNet 直接返回 List[(B, L), ...]
            est_list = model(mix_wav)  # 长度 = n_srcs， 每个 (B, L)
            # 拼成 (B, C, L)
            est_wavs = torch.stack(est_list, dim=1)

            src = torch.stack([s1_wav, s2_wav], dim=1)  # (B, 2, L)
            loss = criterion(mix_wav.unsqueeze(1).repeat(1,2,1), src, est_wavs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % config['log_interval'] == 0:
                avg = total_loss / config['log_interval']
                print(f"[Epoch {epoch} | Batch {i+1}/{len(train_loader)}] loss = {avg:.4f}")
                total_loss = 0.0

        # —— 验证一轮 —— #
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in cv_loader:
                mix_wav = batch['mix_wav'].to(device)
                s1_wav  = batch['s1_wav'].to(device)
                s2_wav  = batch['s2_wav'].to(device)

                est_list = model(mix_wav)
                est_wavs = torch.stack(est_list, dim=1)
                src = torch.stack([s1_wav, s2_wav], dim=1)
                loss = criterion(mix_wav.unsqueeze(1).repeat(1,2,1), src, est_wavs)
                val_loss += loss.item()
        val_loss /= len(cv_loader)
        print(f"Epoch {epoch} 验证集损失: {val_loss:.4f}")
        scheduler.step(val_loss)

        # 保存 checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(config['ckpt_dir'], f"tfgridnet_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"→ 新最佳模型已保存到 {ckpt_path}")

        if epoch % config['save_interval'] == 0:
            ckpt_latest = os.path.join(config['ckpt_dir'], f"tfgridnet_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt_latest)

if __name__ == "__main__":
    main()
