from functools import partial
import dgl
import torch

from src.config.nn_config import VERY_LARGE_NUMBER
from src.config.graph_config import NODE_ENEMY, NODE_ALLY


def get_batched_index(batched_graph, index_list, return_num_targets=False):
    _num_nodes = 0
    return_indices = []
    if return_num_targets:
        num_targets = []

    for num_node, target_index in zip(batched_graph.batch_num_nodes, index_list):
        indices = [i + _num_nodes for i in target_index]
        return_indices.extend(indices)
        _num_nodes += num_node
        if return_num_targets:
            num_targets.append(len(target_index))

    if not return_num_targets:
        return return_indices
    else:
        return return_indices, num_targets


def pop_node_feature_dict(graph, node_feature_key='node_feature'):
    ret_dict = dict()
    for ntype in graph.ntypes:
        ret_dict[ntype] = graph.nodes[ntype].data.pop(node_feature_key)
    return ret_dict


def set_node_feature_dict(graph, feature_dict, node_feature_key='node_feature'):
    for key, val in feature_dict.items():
        graph.nodes[key].data[node_feature_key] = val


def filter_by_edge_type_idx(edges, etype_idx):
    return edges.data['edge_type'] == etype_idx


def get_filtered_edge_index_by_type(graph, etype_idx):
    filter_func = partial(filter_by_edge_type_idx, etype_idx=etype_idx)
    edge_idx = graph.filter_edges(filter_func)
    return edge_idx


def filter_by_node_type_idx(nodes, ntype_idx):
    return nodes.data['node_type'] == ntype_idx


def get_filtered_node_index_by_type(graph, ntype_idx):
    filter_func = partial(filter_by_node_type_idx, ntype_idx=ntype_idx)
    node_idx = graph.filter_nodes(filter_func)
    return node_idx


def filter_by_node_assignment(nodes, assignment):
    return nodes.data['assignment'] == assignment


def get_filtered_node_index_by_assignment(graph, assignment):
    filter_func = partial(filter_by_node_assignment, assignment=assignment)
    node_idx = graph.filter_nodes(filter_func)
    return node_idx


def filter_by_node_index_by_type_and_assignment(nodes, ntype_idx, assignment):
    node_type_cond = nodes.data['node_type'] == ntype_idx
    assignment_cond = nodes.data['assignment'] == assignment
    return node_type_cond * assignment_cond


def get_filtered_node_index_by_type_and_assignment(graph, ntype_idx, assignment):
    filter_func = partial(filter_by_node_index_by_type_and_assignment,
                          ntype_idx=ntype_idx, assignment=assignment)
    node_idx = graph.filter_nodes(filter_func)
    return node_idx


def get_largest_number_of_enemy_nodes(graphs):
    max_num_enemy = 0
    for graph in graphs:
        num_enemy = len(get_filtered_node_index_by_type(graph, NODE_ENEMY))
        if max_num_enemy <= num_enemy:
            max_num_enemy = num_enemy
    return max_num_enemy


def get_number_of_ally_nodes(graphs):
    nums_ally = []
    for graph in graphs:
        num_ally = len(get_filtered_node_index_by_type(graph, NODE_ALLY))
        nums_ally.append(num_ally)
    return nums_ally


# ************************ WARNING ****************************
# DO NOT ERASE UNUSED ARGUMENT 'id_range' OF curie_initializer
# ************************ WARNING ****************************

def curie_initializer(shape, dtype, ctx, id_range):
    return torch.ones(shape, dtype=dtype, device=ctx) * - VERY_LARGE_NUMBER


# ************************ WARNING ****************************
# DO NOT ERASE UNUSED ARGUMENT 'id_range' OF curie_initializer
# ************************ WARNING ****************************


def _get_index_mapper_list(graph, next_graph, cur_node_start_idx, next_node_start_idx):
    """
    generate map tags in the next graph into the tags in the graph if exist.
    """
    cur_idx = []
    next_idx = []

    next_graph_tag = next_graph.ndata['tag']
    ally_node_index = get_filtered_node_index_by_type(graph, NODE_ALLY)
    for cn_index in graph.nodes()[ally_node_index]:
        curr_tag = graph.ndata['tag'][cn_index]
        next_graph_index = (next_graph_tag == curr_tag).nonzero().squeeze().int()
        if next_graph_index.nelement() == 0:
            pass
        elif next_graph_index.nelement() == 1:
            cur_idx.append((cn_index + cur_node_start_idx).tolist())
            next_idx.append((next_graph_index + next_node_start_idx).tolist())
        else:
            raise RuntimeError("Existing multiple units with same tag in next graph")
    return cur_idx, next_idx


def get_index_mapper(graph, next_graph):
    """
    :param graph:
    :param next_graph:
    :return:
    """
    if type(graph) == dgl.BatchedDGLGraph and type(next_graph) == dgl.BatchedDGLGraph:
        graphs = dgl.unbatch(graph)
        next_graphs = dgl.unbatch(next_graph)

        cur_idx = []
        next_idx = []

        curr_num_nodes = 0
        next_num_nodes = 0
        for g, ng in zip(graphs, next_graphs):
            _curr_num_nodes = len(get_filtered_node_index_by_type(g, NODE_ALLY))
            _next_num_nodes = len(get_filtered_node_index_by_type(ng, NODE_ALLY))
            ci, ni = _get_index_mapper_list(g, ng, curr_num_nodes, next_num_nodes)
            cur_idx.extend(ci)
            next_idx.extend(ni)
            curr_num_nodes += _curr_num_nodes
            next_num_nodes += _next_num_nodes
    else:
        cur_idx, next_idx = _get_index_mapper_list(graph, next_graph, 0, 0)

    return cur_idx, next_idx


def minus_large_num_initializer(shape, dtype, ctx, id_range):
    return torch.ones(shape, dtype=dtype, device=ctx) * - VERY_LARGE_NUMBER
