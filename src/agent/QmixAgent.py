import torch

from src.rl.Qnet import Qnet, QnetConfig
from src.rl.QmixNetwork import QmixNetwork, QmixerConfig
from src.brain.QmixBrain import QmixBrain, QmixBrainConfig

from src.config.ConfigBase_refac import ConfigBase
from src.memory.MemoryBase import NstepMemoryConfig


class QmixAgentConfig(ConfigBase):
    def __init__(self, name='qmixagnet', qnet_conf=None, mixer_conf=None, brain_conf=None, fit_conf=None, buffer_conf=None):

        super(QmixAgentConfig, self).__init__(name=name)
        self.qnet = QnetConfig() if qnet_conf is None else qnet_conf
        self.mixer = QmixerConfig() if mixer_conf is None else mixer_conf
        self.brain = QmixBrainConfig() if brain_conf is None else brain_conf
        self.fit = {'batch_size': 256} if fit_conf is None else fit_conf
        self.buffer = NstepMemoryConfig() if buffer_conf is None else buffer_conf


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

    # def get_action(self, inputs):


if __name__ == '__main__':
    conf = QmixAgentConf()
    #     # conf()

    rconf = rQmixAgentConf()
    rconf()
    print(conf())
    print(rconf())