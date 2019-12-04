import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nn.MLP import MultiLayerPerceptron as MLP


class RelationalGraphLayer(nn.Module):
    """
    Relational Graph Network layer
    """

    def __init__(self,
                 input_node_dim: int,
                 output_node_dim: int,
                 node_types: list,
                 edge_types: list,
                 updater_conf: dict,
                 use_residual: bool = False,
                 use_concat: bool = False):
        """
        :param input_node_dim:
        :param output_node_dim:
        :param node_types:
        :param edge_types:
        """

        super(RelationalGraphLayer, self).__init__()

        self.input_dim = input_node_dim
        self.output_dim = output_node_dim
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
            edge_updater_input_dim = input_node_dim * 2

            # node updater's input : [ node_feat || #. edge types * node_feat ||  init_node_feat ]
            node_updater_input_dim = input_node_dim * (len(edge_types) + 2)

        else:
            # edge updater's input : [ node_feat ]
            edge_updater_input_dim = input_node_dim

            # node updater's input : [ node_feat || #. edge types * node_feat ]
            node_updater_input_dim = input_node_dim * (len(edge_types) + 1)

        self.edge_updater_input_dim = edge_updater_input_dim
        self.node_updater_input_dim = node_updater_input_dim

        # initialize node updaters
        updater_conf['input_dimension'] = node_updater_input_dim
        updater_conf['output_dimension'] = output_node_dim

        self.node_updaters = nn.ModuleDict()
        for ntype_idx in node_types:
            node_updater = MLP(**updater_conf)
            self.node_updaters[ntype_idx] = node_updater

        # initialize edge updaters
        updater_conf['input_dimension'] = edge_updater_input_dim
        updater_conf['output_dimension'] = output_node_dim

        self.edge_updaters = nn.ModuleDict()
        for etype_idx in edge_types:
            edge_updater = MLP(**updater_conf)
            self.edge_updaters[etype_idx] = edge_updater

    def forward(self, graph, node_feature):
        if self.use_concat:
            graph.ndata['node_feature'] = torch.cat([node_feature, graph.ndata['init_node_feature']], dim=1)
        else:
            graph.ndata['node_feature'] = node_feature

    def message_function(self, edges):
        src_node_features = edges.src['node_feature']
        edge_types = edges.data['edge_type']

        device = src_node_features.device

        msg_dict = dict()
        for i in self.edge_types:
            msg = torch.zeros(src_node_features.shape[0], self.edge_updater_input_dim, device=device)
            updater = self.edge_updaters[i]

            curr_relation_mask = edge_types == i
            curr_relation_pos = torch.arange(src_node_features.shape[0])[curr_relation_mask]
            if curr_relation_mask.sum() == 0:
                msg_dict['msg_{}'.format(i)] = msg
            else:
                curr_node_features = src_node_features[curr_relation_mask]
                msg[curr_relation_pos, :] = F.relu(updater(curr_node_features))
                msg_dict['msg_{}'.format(i)] = msg
        return msg_dict

    def reduce_function(self, nodes, update_edge_type_indices):
        node_feature = nodes.data['node_feature']
        device = node_feature.device

        node_enc_input = torch.zeros(node_feature.shape[0], self.node_updater_input_dim, device=device)
        if self.use_concat:
            node_enc_input[:, :self.model_dim * 2] = F.relu(node_feature)
            start_index = 2
        else:
            node_enc_input[:, :self.model_dim] = F.relu(node_feature)
            start_index = 1

        for i in update_edge_type_indices:
            msg = nodes.mailbox['msg_{}'.format(i)]
            reduced_msg = msg.sum(dim=1)
            node_enc_input[:, self.model_dim * (i + start_index):self.model_dim * (i + start_index + 1)] = reduced_msg

        return {'aggregated_node_feature': node_enc_input}

