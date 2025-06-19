# config.py
config = {
    'sample_rate'    : 16000,
    'n_fft'          : 512,
    'hop_length'     : 128,
    'win_length'     : 512,
    'enc_dim'        : 256,
    'grid_units'     : [3, 3, 3],
    'n_speakers'     : 2,
    'batch_size'     : 4,
    'epochs'         : 100,
    'lr'             : 1e-3,
    'device'         : 'cuda',      # 或 'cuda:0'
    'ckpt_dir'       : 'checkpoints',
    'log_interval'   : 100,         # 每训练多少 batch 打印一次
    'save_interval'  : 1,           # 每多少 epoch 保存一次模型
}
