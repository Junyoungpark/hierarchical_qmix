import torch

from src.rl.Qnet import Qnet, QnetConfig, rQnetConfig
from src.rl.QmixNetwork import QmixNetwork, QmixerConfig
from src.brain.QmixBrain import QmixBrain, QmixBrainConfig, rQmixBrainConfig

from src.config.ConfigBase_refac import ConfigBase as rConfigBase
from src.config.ConfigBase import ConfigBase
from src.memory.MemoryBase import NstepMemoryConfig


class QmixAgentConfig(ConfigBase):
    def __init__(self, qnet_conf=None, mixer_conf=None, brain_conf=None, fit_conf=None, buffer_conf=None):

        super(QmixAgentConfig, self).__init__(qnet=qnet_conf, mixer=mixer_conf, brain=brain_conf, fit=fit_conf,
                                              buffer=buffer_conf)
        self.qnet = QnetConfig()
        self.mixer = QmixerConfig()
        self.brain = QmixBrainConfig()
        self.fit = {'prefix': 'agent-fit',
                    'batch_size': 256
                    }
        self.buffer = NstepMemoryConfig()


class rQmixAgentConf(rConfigBase):
    def __init__(self, name='qmix-agent', qnet_conf=None, brain_conf=None):
        super(rQmixAgentConf, self).__init__(name=name)

        self.qnet = rQnetConfig() if qnet_conf is None else qnet_conf
        self.qnet.move_module['input_dimension'] = 128

        self.brain = rQmixBrainConfig() if brain_conf is None else brain_conf


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
    # conf()

    rconf = rQmixAgentConf()
    rconf()
    print(conf())
    print(rconf())