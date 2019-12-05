import torch
import torch.nn as nn

from src.optim.Lookahead import Lookahead
from src.optim.RAdam import RAdam


class BrainBase(nn.Module):
    def __init__(self):
        super(BrainBase, self).__init__()

    @staticmethod
    def update_target_network(tau, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def clip_and_optimize(optimizer, parameters, loss, clip_val=None, scheduler=None):
        optimizer.zero_grad()
        loss.backward()
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(parameters, clip_val)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    @staticmethod
    def _get_optimizer(target_opt):
        if target_opt in ['Adam', 'adam']:
            opt = torch.optim.Adam
        elif target_opt in ['Radam', 'RAdam', 'radam']:
            opt = RAdam
        elif target_opt in ['lookahead']:
            opt = Lookahead
        else:
            raise RuntimeError("Not supported optimizer type: {}".format(target_opt))
        return opt

    def set_optimizer(self, target_opt, lr, parameters):
        opt = self._get_optimizer(target_opt)
        if target_opt == 'lookahead':
            base_opt = RAdam(params=parameters, lr=lr)
            optimizer = opt(base_opt)
        else:
            optimizer = opt(params=parameters, lr=lr)

        return optimizer
