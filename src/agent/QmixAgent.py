from copy import deepcopy

import dgl
import torch

from src.rl.MultiStepQnet import MultiStepQnet, MultiStepQnetConfig
from src.rl.QmixNetwork import QmixNetwork, QmixNetworkConfig
from src.brain.QmixBrain import QmixBrain, QmixBrainConfig
from src.util.sc2_util import nn_action_to_sc2_action
from src.util.graph_util import get_largest_number_of_enemy_nodes

from src.config.ConfigBase import ConfigBase
from src.memory.MemoryBase import NstepMemoryConfig, NstepMemory


class QmixAgentConfig(ConfigBase):
    def __init__(self, name='qmixagnet', qnet_conf=None, mixer_conf=None, brain_conf=None, fit_conf=None,
                 buffer_conf=None):
        super(QmixAgentConfig, self).__init__(name=name, qnet=qnet_conf, mixer=mixer_conf, brain=brain_conf,
                                              fit=fit_conf, buffer=buffer_conf)
        self.qnet = MultiStepQnetConfig()
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

        qnet = MultiStepQnet(deepcopy(self.qnet_conf))
        mixer = QmixNetwork(deepcopy(self.mixer_conf))
        qnet2 = MultiStepQnet(deepcopy(self.qnet_conf))
        mixer2 = QmixNetwork(deepcopy(self.mixer_conf))

        self.brain = QmixBrain(conf=self.brain_conf, qnet=qnet, mixer=mixer, qnet2=qnet2, mixer2=mixer2)
        self.buffer = NstepMemory(self.buffer_conf)

    def get_action(self, hist_graph, curr_graph, tag2unit_dict):
        assert isinstance(curr_graph, dgl.DGLGraph), "get action is designed to work on a single graph!"
        num_time_steps = hist_graph.batch_size
        hist_node_feature = hist_graph.ndata.pop('node_feature')
        curr_node_feature = curr_graph.ndata.pop('node_feature')
        maximum_num_enemy = get_largest_number_of_enemy_nodes([curr_graph])

        nn_actions, info_dict = self.brain.get_action(num_time_steps=num_time_steps,
                                                      hist_graph=hist_graph,
                                                      hist_feature=hist_node_feature,
                                                      curr_graph=curr_graph,
                                                      curr_feature=curr_node_feature,
                                                      maximum_num_enemy=maximum_num_enemy)

        ally_tags = info_dict['ally_tags']
        enemy_tags = info_dict['enemy_tags']

        sc2_actions = nn_action_to_sc2_action(nn_actions=nn_actions,
                                              ally_tags=ally_tags,
                                              enemy_tags=enemy_tags,
                                              tag2unit_dict=tag2unit_dict)

        hist_graph.ndata['node_feature'] = hist_node_feature
        curr_graph.ndata['node_feature'] = curr_node_feature
        return nn_actions, sc2_actions, info_dict

    def fit(self):
        pass


if __name__ == '__main__':
    conf = QmixAgentConfig()
    agent = QmixAgent(conf)
