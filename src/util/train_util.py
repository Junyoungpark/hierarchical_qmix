import torch


def dn(tensor: torch.Tensor):
    return tensor.cpu().detach().numpy()
