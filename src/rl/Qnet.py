import torch
import torch.nn as nn

from src.rl.ActionModules import MoveModule, AttackModule
from src.utils.graph_utils import get_filtered_node_index_by_type
from src.config.ConfigBase import ConfigBase
from src.config.graph_config import NODE_ALLY, EDGE_ENEMY
from src.nn.MLP import MLPConfig


class QnetConfig(ConfigBase):

    def __init__(self,
                 qnet_conf,
                 move_module_conf,
                 attack_module_conf):
        super(QnetConfig, self).__init__(qnet=qnet_conf,
                                         move_module=move_module_conf,
                                         attack_module=attack_module_conf)

        mlp_conf = MLPConfig().mlp

        self.qnet = {
            'prefix': 'qnet',
            'attack_edge_type_index': EDGE_ENEMY,
            'ally_node_type_index': NODE_ALLY
        }

        self.move_module = {'prefix': 'qnet-move'}.update(mlp_conf)
        self.attack_module = {'prefix': 'qnet-attack'}.update(mlp_conf)


class Qnet(nn.Module):

    def __init__(self,
                 conf):
        super(Qnet, self).__init__()

        self.conf = conf
        self.move_module = MoveModule(self.conf.move_module_conf)
        self.attack_module = AttackModule(self.conf.attack_module_conf)

    def forward(self, graph, node_feature, maximum_num_enemy):
        # compute move qs
        move_argument = self.move_module(graph, node_feature)

        # compute attack qs
        attack_edge_type_index = self.conf.qnet_conf['attack_edge_type_index']
        attack_argument = self.attack_module(graph, node_feature, maximum_num_enemy, attack_edge_type_index)

        return move_argument, attack_argument

    def compute_qs(self, graph, node_feature, maximum_num_enemy):

        # get qs of actions
        move_arg, attack_arg = self(graph, node_feature, maximum_num_enemy)

        qs = torch.cat((move_arg, attack_arg), dim=-1)  # of all units including enemies

        ally_node_type_index = self.conf.qnet_conf['ally_node_type_index']
        ally_node_indices = get_filtered_node_index_by_type(graph, ally_node_type_index)
        qs = qs[ally_node_indices, :]  # of only ally units

        ally_tags = graph.ndata['tag']
        ally_tags = ally_tags[ally_node_indices]
        if 'enemy_tag' in graph.ndata.keys():
            enemy_tags = graph.ndata['enemy_tag']
        else:
            enemy_tags = torch.zeros_like(ally_tags).view(-1, 1)  # dummy

        enemy_tags = enemy_tags[ally_node_indices, :]

        return_dict = dict()
        # for RL training
        return_dict['qs'] = qs

        # for SC2 interfacing
        return_dict['ally_tags'] = ally_tags
        return_dict['enemy_tags'] = enemy_tags
        return return_dict
