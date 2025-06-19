import torch.nn as nn

def get_activation_layer(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    elif name == "prelu":
        return nn.PReLU
    elif name == "elu":
        return nn.ELU
    elif name == "gelu":
        return nn.GELU
    elif name == "tanh":
        return nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {name}")
