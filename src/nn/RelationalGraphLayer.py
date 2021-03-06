import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.MLP import MultiLayerPerceptron as MLP
from src.util.graph_util import get_filtered_node_index_by_type


class RelationalGraphLayer(nn.Module):
    """
    Relational Graph Network layer
    """

    def __init__(self,
                 input_node_dim: int,
                 output_node_dim: int,
                 init_node_dim: int,
                 node_types: list,
                 edge_types: list,
                 updater_conf: dict,
                 use_residual: bool,
                 use_concat: bool):
        """
        :param input_node_dim: input node dim (exclude concat dim)
        :param output_node_dim: output node dim
        :param node_types: list of integer node types
        :param edge_types: list of integer edge types
        :param updater_conf:
        :param use_residual:
        :param use_concat:
        """

        super(RelationalGraphLayer, self).__init__()

        self.input_dim = input_node_dim
        self.output_dim = output_node_dim
        self.init_dim = init_node_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.use_residual = use_residual
        self.use_concat = use_concat

        # assert input dim == output dim when residual hook is true.
        if self.use_residual:
            assert input_node_dim == output_node_dim, "If use_residual, 'input_dim' and 'output_dim' must be equal."

        # infer inter-layer hook type
        if int(self.use_residual) + int(self.use_concat) >= 2:
            warnings.warn("Either one of 'use_residual' or 'use_concat' can be true. 'use_residual' set to be false.")
            self.use_residual = False

        # infer input dimensions for node updaters and edge updaters
        if use_concat:
            # edge updater's input : [ node_feat || init_node_feat ]
            edge_updater_input_dim = input_node_dim + init_node_dim

            # node updater's input : [ node_feat || #. edge types * node_feat ||  init_node_feat ]
            node_updater_input_dim = input_node_dim + output_node_dim * len(edge_types) + init_node_dim

        else:
            # edge updater's input : [ node_feat ]
            edge_updater_input_dim = input_node_dim

            # node updater's input : [ node_feat || #. edge types * node_feat ]
            node_updater_input_dim = input_node_dim + output_node_dim * len(edge_types)

        self.edge_updater_input_dim = edge_updater_input_dim
        self.node_updater_input_dim = node_updater_input_dim

        # initialize node updaters
        updater_conf['input_dimension'] = node_updater_input_dim
        updater_conf['output_dimension'] = output_node_dim

        self.node_updater = nn.ModuleDict()
        for ntype_idx in node_types:
            node_updater = MLP(**updater_conf)
            self.node_updater['updater{}'.format(ntype_idx)] = node_updater

        # initialize edge updaters
        updater_conf['input_dimension'] = edge_updater_input_dim
        updater_conf['output_dimension'] = output_node_dim

        self.edge_updater = nn.ModuleDict()
        for etype_idx in edge_types:
            edge_updater = MLP(**updater_conf)
            self.edge_updater['updater{}'.format(etype_idx)] = edge_updater

    def forward(self, graph, node_feature):
        if self.use_concat:
            graph.ndata['node_feature'] = torch.cat([node_feature, graph.ndata['init_node_feature']], dim=1)
        else:
            graph.ndata['node_feature'] = node_feature

        graph.send_and_recv(graph.edges(),
                            message_func=self.message_function,
                            reduce_func=self.reduce_function)

        for ntype_idx in self.node_types:
            node_indices = get_filtered_node_index_by_type(graph, ntype_idx)
            node_updater = self.node_updater['updater{}'.format(ntype_idx)]
            apply_node_func = partial(self.apply_node_function_multi_type, updater=node_updater)
            graph.apply_nodes(apply_node_func, v=node_indices)

        updated_node_feature = graph.ndata.pop('updated_node_feature')
        _ = graph.ndata.pop('aggregated_node_feature')
        _ = graph.ndata.pop('node_feature')

        if self.use_residual:
            updated_node_feature = updated_node_feature + node_feature

        return updated_node_feature

    def message_function(self, edges):
        src_node_features = edges.src['node_feature']
        edge_types = edges.data['edge_type']

        device = src_node_features.device

        msg_dict = dict()
        for i in self.edge_types:
            msg = torch.zeros(src_node_features.shape[0], self.output_dim, device=device)
            updater = self.edge_updater['updater{}'.format(i)]

            curr_relation_mask = edge_types == i
            curr_relation_pos = torch.arange(src_node_features.shape[0])[curr_relation_mask]
            if curr_relation_mask.sum() == 0:
                msg_dict['msg_{}'.format(i)] = msg
            else:
                curr_node_features = src_node_features[curr_relation_mask]
                msg[curr_relation_pos, :] = F.relu(updater(curr_node_features))
                msg_dict['msg_{}'.format(i)] = msg
        return msg_dict

    def reduce_function(self, nodes):
        node_feature = nodes.data['node_feature']
        device = node_feature.device

        node_enc_input = torch.zeros(node_feature.shape[0],
                                     self.node_updater_input_dim,
                                     device=device)

        if self.use_concat:
            node_enc_input[:, :self.input_dim + self.init_dim] = F.relu(node_feature)
            start_index = self.input_dim + self.init_dim
        else:
            node_enc_input[:, :self.input_dim] = F.relu(node_feature)
            start_index = self.input_dim

        for i in self.edge_types:
            msg = nodes.mailbox['msg_{}'.format(i)]
            reduced_msg = msg.sum(dim=1)
            node_enc_input[:, start_index + i * self.output_dim:
                              start_index + (i+1) * self.output_dim] = reduced_msg

        return {'aggregated_node_feature': node_enc_input}

    def apply_node_function_multi_type(self, nodes, updater):
        aggregated_node_feature = nodes.data['aggregated_node_feature']
        out = updater(aggregated_node_feature)
        return {'updated_node_feature': out}
