import torch
import torch.nn as nn

from src.nn.MultiStepInputGraphNetwork import MultiStepInputGraphNetwork, MultiStepInputGraphNetworkConfig
from src.rl.Qnet import Qnet, QnetConfig
from src.config.ConfigBase import ConfigBase


class MultiStepQnetConfig(ConfigBase):
    def __init__(self, name='multistepqnet', qnet_config=None, hist_enc_conf=None):
        super(MultiStepQnetConfig, self).__init__(name=name, qnet=qnet_config, hist_enc=hist_enc_conf)
        self.qnet = QnetConfig()
        self.hist = MultiStepInputGraphNetworkConfig()


class MultiStepQnet(nn.Module):

    def __init__(self, conf):
        super(MultiStepQnet, self).__init__()
        self.encoder = MultiStepInputGraphNetwork(conf.hist)
        self.qnet = Qnet(conf.qnet)

    def forward(self,
                num_time_steps,
                hist_graph, hist_feature,
                curr_graph, curr_feature, maximum_num_enemy):
        hist_current_encoded_node_feature = self.encoder(num_time_steps,
                                                         hist_graph,
                                                         hist_feature,
                                                         curr_graph,
                                                         curr_feature)

        move_arg, attack_arg = self.qnet(graph=curr_graph,
                                         node_feature=hist_current_encoded_node_feature,
                                         maximum_num_enemy=maximum_num_enemy)

        return move_arg, attack_arg

    def compute_qs(self,
                   num_time_steps,
                   hist_graph, hist_feature,
                   curr_graph, curr_feature, maximum_num_enemy,
                   mixer=None):
        hist_current_encoded_node_feature = self.encoder(num_time_steps,
                                                         hist_graph,
                                                         hist_feature,
                                                         curr_graph,
                                                         curr_feature)

        original_hidden_feat = hist_current_encoded_node_feature
        if mixer is not None:
            _, _, normed_group_dot_prd = mixer.submixer.get_feat(curr_graph, hist_current_encoded_node_feature)
            hist_current_encoded_node_feature = torch.cat([hist_current_encoded_node_feature,
                                                           normed_group_dot_prd],
                                                          dim=1)

        q_dict = self.qnet.compute_qs(curr_graph,
                                      hist_current_encoded_node_feature,
                                      maximum_num_enemy)

        q_dict['hidden_feat'] = original_hidden_feat
        return q_dict

    def get_action(self,
                   num_time_steps,
                   hist_graph, hist_feature,
                   curr_graph, curr_feature, maximum_num_enemy,
                   eps, mixer=None):
        q_dict = self.compute_qs(num_time_steps,
                                 hist_graph, hist_feature,
                                 curr_graph, curr_feature, maximum_num_enemy, mixer)

        nn_actions, q_dict = self.qnet.get_action_from_q(q_dict=q_dict, eps=eps)
        return nn_actions, q_dict
