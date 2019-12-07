import torch

from src.rl.Qnet import Qnet, QnetConfig
from src.rl.QmixNetwork import QmixNetwork, QmixNetworkConfig
from src.brain.QmixBrain import QmixBrain, QmixBrainConfig

from src.config.ConfigBase import ConfigBase
from src.memory.MemoryBase import NstepMemoryConfig, NstepMemory


class QmixAgentConfig(ConfigBase):
    def __init__(self, name='qmixagnet', qnet_conf=None, mixer_conf=None, brain_conf=None, fit_conf=None,
                 buffer_conf=None):
        super(QmixAgentConfig, self).__init__(name=name, qnet=qnet_conf, mixer=mixer_conf, brain=brain_conf,
                                              fit=fit_conf, buffer=buffer_conf)
        self.qnet = QnetConfig()
        self.mixer = QmixNetworkConfig()
        self.brain = QmixBrainConfig()
        self.fit = {'batch_size': 256}
        self.buffer = NstepMemoryConfig()


class QmixAgent(torch.nn.Module):
    def __init__(self, conf):
        super(QmixAgent, self).__init__()
        self.qnet_conf = conf.qnet
        self.mixer_conf = conf.mixer
        self.brain_conf = conf.brain

        self.fit_conf = conf.fit
        self.buffer_conf = conf.buffer

        qnet = Qnet(self.qnet_conf)
        mixer = QmixNetwork(self.mixer_conf)
        qnet2 = Qnet(self.qnet_conf)
        mixer2 = QmixNetwork(self.mixer_conf)

        self.brain = QmixBrain(conf=self.brain_conf, qnet=qnet, mixer=mixer, qnet2=qnet2, mixer2=mixer2)
        self.buffer = NstepMemory(self.buffer_conf)

    def get_action(self, inputs):
        pass

    def fit(self, inputs):
        pass




if __name__ == '__main__':
    conf = QmixAgentConfig()
    agent = QmixAgent(conf)

