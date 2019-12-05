import dgl
import torch

from src.rl.Qnet import Qnet
from src.rl.Qmixer import QMixer
from src.brain.QmixBrain import QmixBrain, QmixBrainConfig

from src.config.ConfigBase import ConfigBase


class QmixAgentConf(ConfigBase):
    def __init__(self):
        super(QmixAgentConf, self).__init__()


class QmixAgent(torch.nn.Module):
    def __init__(self, conf, qnet_conf, mixer_conf):
        super(QmixAgent, self).__init__()

        qnet = None
        mixer = None
        qnet2 = None
        mixer2 = None

        self.brain = QmixBrain