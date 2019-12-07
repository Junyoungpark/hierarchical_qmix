import dgl
import itertools
import torch
from sc2.game_state import GameState
from src.config.graph_config import NODE_ALLY, NODE_ENEMY, EDGE_ALLY, EDGE_ENEMY, EDGE_IN_ATTACK_RANGE, \
    EDGE_ALLY_TO_ENEMY, NUM_EDGE_TYPES, NUM_NODE_TYPES
from src.util.graph_util import minus_large_num_initializer
from src.util.sc2_config import NUM_TOTAL_TYPES, type2onehot
import numpy as np


def process_game_state_to_dgl(game_state: GameState):
    units = game_state.units

    ally_units = units.owned
    enemy_units = units.enemy

    num_allies = len(ally_units)
    num_enemies = len(enemy_units)

    exist_allies = False
    node_types = []

    g = dgl.DGLGraph(multigraph=True)
    g.set_e_initializer(dgl.init.zero_initializer)

    # using minus_large_num_initializer for node features matters a lot !
    # working as a mask for computing action probs later.
    g.set_n_initializer(minus_large_num_initializer)

    node_features = []

    allies_health = 0
    allies_health_percentage = 0
    ally_indices = []

    tags = [unit.tag for unit in ally_units + enemy_units]
    tags_tensor = torch.LongTensor(tags)

    tag2unit_dict = dict()

    if num_allies >= 1:
        exist_allies = True
        allies_center_pos = ally_units.center
        allies_unit_dict = dict()
        allies_index_dict = dict()
        for i, allies_unit in enumerate(ally_units):
            tag2unit_dict[allies_unit.tag] = allies_unit
            ally_indices.append(i)
            node_feature = list()
            one_hot_type_id = get_one_hot_unit_type(allies_unit.type_id.value)
            node_feature.extend(one_hot_type_id)
            node_feature.extend(list(allies_center_pos - allies_unit.position))
            # Absolute positoin
            node_feature.extend(list(allies_unit.position))
            node_feature.append(allies_unit.health_max)
            node_feature.append(allies_unit.health_percentage)
            node_feature.append(allies_unit.weapon_cooldown)
            node_feature.append(allies_unit.ground_dps)
            one_hot_node_type = get_one_hot_node_type(NODE_ALLY)
            node_feature.extend(one_hot_node_type)
            node_features.append(node_feature)
            allies_unit_dict[allies_unit] = i
            allies_index_dict[i] = allies_unit
            node_types.append(NODE_ALLY)
            allies_health += allies_unit.health
            allies_health_percentage += allies_unit.health_percentage

    enemies_health = 0
    enemies_health_percentage = 0
    enemies_indices = []

    if num_enemies >= 1:
        enemy_center_pos = enemy_units.center
        enemy_unit_dict = dict()
        enemy_index_dict = dict()
        for j, enemy_unit in enumerate(enemy_units):
            tag2unit_dict[enemy_unit.tag] = enemy_unit
            enemies_indices.append(num_allies + j)
            node_feature = list()
            one_hot_type_id = get_one_hot_unit_type(enemy_unit.type_id.value)
            node_feature.extend(one_hot_type_id)
            node_feature.extend(list(enemy_center_pos - enemy_unit.position))
            # absolute position
            node_feature.extend(list(enemy_unit.position))
            node_feature.append(enemy_unit.health_max)
            node_feature.append(enemy_unit.health_percentage)
            node_feature.append(enemy_unit.weapon_cooldown)
            node_feature.append(enemy_unit.ground_dps)
            one_hot_node_type = get_one_hot_node_type(NODE_ENEMY)
            node_feature.extend(one_hot_node_type)
            node_features.append(node_feature)
            enemy_unit_dict[enemy_unit] = j + num_allies
            enemy_index_dict[j + num_allies] = enemy_unit
            node_types.append(NODE_ENEMY)
            enemies_health += enemy_unit.health
            enemies_health_percentage += enemy_unit.health_percentage

    if num_allies + num_enemies >= 1:
        node_features = np.stack(node_features)  # [Num total units x Num features]
        node_features = torch.Tensor(node_features)

        node_types = torch.Tensor(node_types).reshape(-1)

        unit_indices = torch.Tensor(ally_indices + enemies_indices).reshape(-1).int()
        num_nodes = node_features.size(0)

    if exist_allies:
        # Add Node features: allies + enemies
        g.add_nodes(num_nodes, {'node_feature': node_features,
                                'node_type': node_types,
                                'tag': tags_tensor,
                                'node_index': unit_indices,
                                'init_node_feature': node_features})

        if num_allies >= 2:
            # Add allies edges
            allies_edge_indices = cartesian_product(ally_indices, ally_indices, return_1d=True)

            # To support hyper network encoder, we keep two edge_types
            allies_edge_type = torch.Tensor(data=(EDGE_ALLY,))
            allies_edge_type_one_hot = torch.Tensor(data=get_one_hot_edge_type(EDGE_ALLY))
            num_allies_edges = len(allies_edge_indices[0])

            g.add_edges(allies_edge_indices[0], allies_edge_indices[1],
                        {'edge_type_one_hot': allies_edge_type_one_hot.repeat(num_allies_edges, 1),
                         'edge_type': allies_edge_type.repeat(num_allies_edges)})

        if num_allies >= 1 and num_enemies >= 1:
            # Constructing bipartite graph for computing primitive attack on attack

            bipartite_edges = cartesian_product(enemies_indices, ally_indices, return_1d=True)

            # the edges from enemies to the allies
            # To support hyper network encoder, we keep two edge_types
            inter_army_edge_type = torch.Tensor(data=(EDGE_ENEMY,))
            inter_army_edge_type_one_hot = torch.Tensor(data=get_one_hot_edge_type(EDGE_ENEMY))
            num_inter_army_edges = len(bipartite_edges[0])

            g.add_edges(bipartite_edges[0], bipartite_edges[1],
                        {'edge_type_one_hot': inter_army_edge_type_one_hot.repeat(num_inter_army_edges, 1),
                         'edge_type': inter_army_edge_type.repeat(num_inter_army_edges)})

            # the edges from allies to the enemies
            inter_army_edge_type = torch.Tensor(data=(EDGE_ALLY_TO_ENEMY,))
            inter_army_edge_type_one_hot = torch.Tensor(data=get_one_hot_edge_type(EDGE_ALLY_TO_ENEMY))
            num_inter_army_edges = len(bipartite_edges[0])

            g.add_edges(bipartite_edges[1], bipartite_edges[0],
                        {'edge_type_one_hot': inter_army_edge_type_one_hot.repeat(num_inter_army_edges, 1),
                         'edge_type': inter_army_edge_type.repeat(num_inter_army_edges)})

            for ally_unit in ally_units:
                # get all in-attack-range units. include allies units
                in_range_units = enemy_units.in_attack_range_of(ally_unit)
                if in_range_units:  # when in-attack-range units exist
                    allies_index = allies_unit_dict[ally_unit]
                    for in_range_unit in in_range_units:
                        enemy_index = enemy_unit_dict[in_range_unit]
                        # Expected bottleneck (2) -> Doubled assignment of edges
                        edge_in_attack_range = torch.Tensor(data=(EDGE_IN_ATTACK_RANGE,))
                        edge_in_attack_range = edge_in_attack_range.reshape(-1)
                        g.add_edge(enemy_index, allies_index, {'edge_type': edge_in_attack_range})
    else:
        pass

    ret_dict = dict()
    ret_dict['g'] = g

    # For interfacing nn action args with sc2 action commends.
    ret_dict['tag2unit_dict'] = tag2unit_dict
    ret_dict['units'] = units

    return ret_dict


def cartesian_product(*iterables, return_1d=False):
    if return_1d:
        xs = []
        ys = []
        for ij in itertools.product(*iterables):
            xs.append(ij[0])
            ys.append(ij[1])
        ret = (xs, ys)

    else:
        ret = [i for i in itertools.product(*iterables)]
    return ret


def get_one_hot_unit_type(unit_type: int):
    ret = [0] * NUM_TOTAL_TYPES
    ret[type2onehot[unit_type]] = 1.0
    return ret


def get_one_hot_node_type(node_type: int):
    ret = [0] * NUM_NODE_TYPES
    ret[node_type] = 1.0
    return ret


def get_one_hot_edge_type(edge_type: int):
    ret = [0] * NUM_EDGE_TYPES
    ret[edge_type] = 1.0
    return ret
