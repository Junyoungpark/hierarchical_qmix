from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from src.rl.ActionModules import MoveModule, AttackModule
from src.util.graph_util import get_filtered_node_index_by_type
from src.config.ConfigBase import ConfigBase
from src.config.graph_config import NODE_ALLY, EDGE_ENEMY
from src.nn.MLP import MLPConfig
from src.config.nn_config import VERY_LARGE_NUMBER

from src.util.train_util import dn


class QnetConfig(ConfigBase):

    def __init__(self, name='qnet', qnet_conf=None, move_module_conf=None, attack_module_conf=None):
        super(QnetConfig, self).__init__(name=name, qnet=qnet_conf, move_module=move_module_conf,
                                         attack_module=attack_module_conf)

        mlp_conf = MLPConfig().mlp

        self.qnet = {'attack_edge_type_index': EDGE_ENEMY,
                     'ally_node_type_index': NODE_ALLY,
                     'exploration_method': 'eps_greedy'}

        self.move_module = deepcopy(mlp_conf)
        self.move_module['normalization'] = 'layer'
        self.move_module['out_activation'] = None

        self.attack_module = deepcopy(mlp_conf)
        self.attack_module['normalization'] = 'layer'
        self.attack_module['out_activation'] = None


class Qnet(nn.Module):

    def __init__(self,
                 conf):
        super(Qnet, self).__init__()
        self.conf = conf
        self.exploration_method = conf.qnet['exploration_method']
        self.move_module = MoveModule(self.conf.move_module)
        self.attack_module = AttackModule(self.conf.attack_module)

    def forward(self, graph, node_feature, maximum_num_enemy):
        # compute move qs
        move_argument = self.move_module(graph, node_feature)

        # compute attack qs
        attack_edge_type_index = self.conf.qnet['attack_edge_type_index']
        attack_argument = self.attack_module(graph, node_feature, maximum_num_enemy, attack_edge_type_index)

        return move_argument, attack_argument

    def compute_qs(self, graph, node_feature, maximum_num_enemy):

        # get qs of actions
        move_arg, attack_arg = self(graph, node_feature, maximum_num_enemy)

        qs = torch.cat((move_arg, attack_arg), dim=-1)  # of all units including enemies
        np_qs = dn(qs)

        ally_node_type_index = self.conf.qnet['ally_node_type_index']
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

    def get_action_from_q(self, q_dict, eps):
        ally_qs = q_dict['qs']

        device = ally_qs.device

        # if 'enemy_tag' in curr_graph.ndata.keys():
        #     _ = curr_graph.ndata.pop('enemy_tag')

        if self.training:
            if self.exploration_method == "eps_greedy":
                if torch.rand(1, device=device) <= eps:
                    sampling_mask = torch.ones_like(ally_qs, device=device)
                    sampling_mask[ally_qs <= -VERY_LARGE_NUMBER] = -VERY_LARGE_NUMBER
                    dist = torch.distributions.categorical.Categorical(logits=sampling_mask)
                    nn_actions = dist.sample()
                else:
                    nn_actions = ally_qs.argmax(dim=1)
            elif self.exploration_method == "clustered_random":
                if torch.rand(1, device=device) <= eps:
                    q_mask = torch.ones_like(ally_qs, device=device)
                    q_mask[ally_qs <= -VERY_LARGE_NUMBER] = 0
                    sampling_mask = generate_hierarchical_sampling_mask(q_mask, use_hold=False)
                    dist = torch.distributions.categorical.Categorical(logits=sampling_mask)
                    nn_actions = dist.sample()
                else:
                    nn_actions = ally_qs.argmax(dim=1)
            elif self.exploration_method == "noisy_net":
                nn_actions = ally_qs.argmax(dim=1)
            else:
                raise RuntimeError("Not admissible exploration methods.")
        else:
            nn_actions = ally_qs.argmax(dim=1)
        return nn_actions, q_dict


def generate_hierarchical_sampling_mask(q_mask, use_hold):
    n_agent = q_mask.shape[0]
    n_clusters = 2  # Consider (Move & Hold) cluster and Attack cluster
    action_start_indices = [0, 5]
    action_end_indices = [5, None]
    can_attacks = (q_mask[:, 5:].sum(1) >= 1)

    if n_agent / n_clusters >= 2.0:
        n_agent_in_attack = torch.randint(low=1, high=int(np.floor(n_agent / n_clusters)), size=(n_clusters - 1,))
        n_agent_in_attack = min(n_agent_in_attack, can_attacks.sum())

        can_attack_agent_indices = can_attacks.nonzero()  # indices of agents who can attack
        should_move_hold = (~can_attacks).nonzero()

        mask = torch.ones_like(q_mask, device=q_mask.device)

        perm = torch.randperm(len(can_attack_agent_indices))
        attack_idx = perm[:n_agent_in_attack]
        move_hold_among_attackable = perm[n_agent_in_attack:]

        attack_agent_idx = can_attack_agent_indices[attack_idx]
        move_hold_agent_idx = torch.cat([can_attack_agent_indices[move_hold_among_attackable],
                                         should_move_hold],
                                        dim=0)

        # mask-out (make 0 prob. to be sampled) for move and hold
        mask[attack_agent_idx, action_start_indices[0]:action_end_indices[0]] = - VERY_LARGE_NUMBER

        # mask-out (make 0 prob. to be sampled) for attack
        mask[move_hold_agent_idx, action_start_indices[1]:action_end_indices[1]] = - VERY_LARGE_NUMBER

        # post process mask to be attack appropriate
        row, col = torch.where(q_mask == 0)
        mask[row, col] = -VERY_LARGE_NUMBER
    else:
        mask = torch.ones_like(q_mask, device=q_mask.device)
        mask[q_mask <= 0] = -VERY_LARGE_NUMBER

    if not use_hold:
        mask[:, 4] = -VERY_LARGE_NUMBER

    return mask
