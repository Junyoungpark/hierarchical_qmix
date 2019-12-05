import torch

from src.rl.Qnet import Qnet, QnetConfig
from src.rl.QmixNetwork import QmixNetwork, QmixerConfig
from src.brain.QmixBrain import QmixBrain, QmixBrainConfig

from src.config.ConfigBase import ConfigBase


class QmixAgentConf(ConfigBase):
    def __init__(self, qnet_conf=None, mixer_conf=None, brain_conf=None, fit_conf=None, buffer_conf=None):
        super(QmixAgentConf, self).__init__(qnet=qnet_conf, mixer=mixer_conf, brain=brain_conf, fit=fit_conf, buffer=buffer_conf)
        self.qnet = QnetConfig()
        self.mixer = QmixerConfig()
        self.brain = QmixBrainConfig()


class QmixAgent(torch.nn.Module):
    def __init__(self, conf):
        super(QmixAgent, self).__init__()
        self.qnet_conf = conf.qnet_conf
        self.mixer_conf = conf.mixer_conf
        self.brain_conf = conf.brain_conf
        self.fit_conf = conf.fit_conf
        self.buffer_conf = conf.buffer_conf

        qnet = Qnet(self.qnet_conf)
        mixer = QmixNetwork(self.mixer_conf)
        qnet2 = Qnet(self.qnet_conf)
        mixer2 = QmixNetwork(self.mixer_conf)

        self.brain = QmixBrain

    def get_action(self, inputs):