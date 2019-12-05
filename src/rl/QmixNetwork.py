import torch
from src.config.ConfigBase import ConfigBase
from src.rl.Qmixer import QMixer, QMixerConfig
from src.nn.MLP import MLPConfig


class QmixNetworkConfig(ConfigBase):
    def __init__(self, submixer_conf=None, supmixer_conf=None):
        super(QmixNetworkConfig, self).__init__(submixer=submixer_conf, supmixer=supmixer_conf)
        self.submixer = QMixerConfig()
        self.supmixer = {'prefix': 'supmixer'}.update(MLPConfig().mlp)


class QmixNetwork(torch.nn.Module):
    def __init__(self, conf):
        super(QmixNetwork, self).__init__()

        self.submixer_conf = conf.submixer()
        self.supmixer_conf = conf.supmixer()

        self.submixer = QMixer(self.submixer_conf)
        self.supmixer_conf
