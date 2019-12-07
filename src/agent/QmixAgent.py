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
        self.fit = {'batch_size': 256,
                    'hist_num_time_steps': 2}
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

        curr_graph.ndata.pop('enemy_tag')

        return nn_actions, sc2_actions, info_dict

    def fit(self, device='cpu'):
        # the prefix 'c' indicates #current# time stamp inputs
        # the prefix 'n' indicates #next# time stamp inputs

        # expected specs:
        # bs = batch_size, nt = hist_num_time_steps
        # 'h_graph' = list of graph lists [[g_(0,0), g_(0,1), ... g_(0,nt)],
        #                                  [g_(1,0), g_(1,1), ..., g_(1,nt)],
        #                                  [g_(2,0), ..., g_(bs, 0), ... g_(bs, nt)]]
        # 'graph' = list of graphs  [g_(0), g_(1), ..., g_(bs)]

        fit_conf = self.fit_conf

        batch_size = fit_conf['batch_size']
        hist_num_time_steps = fit_conf['hist_num_time_steps']

        c_h_graph, c_graph, actions, rewards, n_h_graph, n_graph, dones = self.buffer.sample(batch_size)

        c_maximum_num_enemy = get_largest_number_of_enemy_nodes(c_graph)
        n_maximum_num_enemy = get_largest_number_of_enemy_nodes(n_graph)

        # batching graphs
        list_c_h_graph = [g for L in c_h_graph for g in L]
        list_n_h_graph = [g for L in n_h_graph for g in L]

        c_hist_graph = dgl.batch(list_c_h_graph)
        n_hist_graph = dgl.batch(list_n_h_graph)

        c_curr_graph = dgl.batch(c_graph)
        n_curr_graph = dgl.batch(n_graph)

        # c_hist_graph.set_n_initializer(curie_initializer)
        # n_hist_graph.set_n_initializer(curie_initializer)
        # c_curr_graph.set_n_initializer(curie_initializer)
        # n_curr_graph.set_n_initializer(curie_initializer)

        # casting actions to one torch tensor
        actions = torch.cat(actions).long()

        # prepare rewards
        rewards = torch.Tensor(rewards)

        # preparing dones
        dones = torch.Tensor(dones)

        if device != 'cpu':
            c_hist_graph.to(torch.device('cuda'))
            n_hist_graph.to(torch.device('cuda'))
            c_curr_graph.to(torch.device('cuda'))
            n_curr_graph.to(torch.device('cuda'))
            actions = actions.to(torch.device('cuda'))
            rewards = rewards.to(torch.device('cuda'))
            dones = dones.to(torch.device('cuda'))

        c_hist_feature = c_hist_graph.ndata.pop('node_feature')
        c_curr_feature = c_curr_graph.ndata.pop('node_feature')

        n_hist_feature = n_hist_graph.ndata.pop('node_feature')
        n_curr_feature = n_curr_graph.ndata.pop('node_feature')

        curr_inputs = {'num_time_steps': hist_num_time_steps,
                       'hist_graph': c_hist_graph,
                       'hist_feature': c_hist_feature,
                       'curr_graph': c_curr_graph,
                       'curr_feature': c_curr_feature,
                       'maximum_num_enemy': c_maximum_num_enemy}

        next_inputs = {'num_time_steps': hist_num_time_steps,
                       'hist_graph': n_hist_graph,
                       'hist_feature': n_hist_feature,
                       'curr_graph': n_curr_graph,
                       'curr_feature': n_curr_feature,
                       'maximum_num_enemy': n_maximum_num_enemy}

        fit_return_dict = self.brain.fit(curr_inputs=curr_inputs,
                                         next_inputs=next_inputs,
                                         actions=actions,
                                         rewards=rewards,
                                         dones=dones)

        return fit_return_dict


if __name__ == '__main__':
    conf = QmixAgentConfig()
    agent = QmixAgent(conf)
