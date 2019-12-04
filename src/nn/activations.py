import torch
import torch.nn.functional as F
import torch.nn as nn

TORCH_ACTIVATION_LIST = ['ReLU',
                         'Sigmoid',
                         'SELU',
                         'leaky_relu',
                         'Softplus']

ACTIVATION_LIST = ['mish', 'swish', None]


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


def get_nn_activation(activation: 'str'):
    if not activation in TORCH_ACTIVATION_LIST + ACTIVATION_LIST:
        raise RuntimeError("Not implemented activation function!")

    if activation in TORCH_ACTIVATION_LIST:
        act = getattr(torch.nn, activation)()

    if activation in ACTIVATION_LIST:
        if activation == 'mish':
            act = Mish()
        elif activation == 'swish':
            act = Swish()
        elif activation is None:
            act = nn.Identity()

    return act
