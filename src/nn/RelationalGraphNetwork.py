import torch.nn as nn
from src.nn.RelationalGraphLayer import RelationalGraphLayer
from src.nn.MLP import MLPConfig

from src.config.ConfigBase import ConfigBase
from src.config.graph_config import (NODE_ALLY, NODE_ENEMY,
                                     EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY)


class RelationalGraphNetworkConfig(ConfigBase):

    def __init__(self, name='gnn', gnn_conf=None):
        super(RelationalGraphNetworkConfig, self).__init__(name=name, gnn=gnn_conf)

        self.gnn = {
            'input_node_dim': 19,
            'hidden_node_dim': 16,
            'output_node_dim': 16,
            'init_node_dim': 19,
            'num_hidden_layers': 1,
            'node_types': [NODE_ALLY, NODE_ENEMY],
            'edge_types': [EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY],
            'updater_conf': MLPConfig().mlp,
            'use_residual': False,
            'use_concat': True,
        }


class RelationalGraphNetwork(nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 hidden_node_dim: int,
                 output_node_dim: int,
                 init_node_dim: int,
                 num_hidden_layers: int,
                 node_types: list,
                 edge_types: list,
                 updater_conf: dict,
                 use_residual: bool,
                 use_concat: bool):

        super(RelationalGraphNetwork, self).__init__()

        self.use_residual = use_residual
        self.use_concat = use_concat

        input_dims = [input_node_dim] + num_hidden_layers * [hidden_node_dim]
        output_dims = num_hidden_layers * [hidden_node_dim] + [output_node_dim]

        self.layers = nn.ModuleList()
        for input_dim, output_dim in zip(input_dims, output_dims):
            use_residual = self.use_residual and input_dim == output_dim

            layer = RelationalGraphLayer(input_node_dim=input_dim,
                                         output_node_dim=output_dim,
                                         init_node_dim=init_node_dim,
                                         node_types=node_types,
                                         edge_types=edge_types,
                                         updater_conf=updater_conf,
                                         use_residual=use_residual,
                                         use_concat=use_concat)
            self.layers.append(layer)

    def forward(self, graph, node_feature):
        for layer in self.layers:
            node_feature = layer(graph, node_feature)
        return node_feature
