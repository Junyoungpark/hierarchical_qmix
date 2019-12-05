import torch.nn as nn
from src.config.ConfigBase import ConfigBase


class QmixBrainConfig(ConfigBase):
    def __init__(self, brain_conf=None, fit_conf=None):
        super(QmixBrainConfig, self).__init__(brain=brain_conf, fit=fit_conf)



class QmixBrain(nn.Module):
    def __init__(self, conf):
        super(QmixBrain, self).__init__()
        self.conf = conf
        self.brain_conf = conf.brain()
        self.fit_conf = conf.fit_conf()


