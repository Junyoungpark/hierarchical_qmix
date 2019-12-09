import torch.nn as nn
from functools import partial

from src.nn.MLP import MLPConfig, MultiLayerPerceptron
from src.util.graph_util import get_filtered_node_index_by_type
from src.config.graph_config import NODE_ALLY

from src.config.ConfigBase import ConfigBase
from src.util.train_util import dn


class FeedForwardConfig(ConfigBase):
    def __init__(self, name='feedforward', mlp_conf=None, ff_conf=None):
        super(FeedForwardConfig, self).__init__(name=name, mlp=mlp_conf, ff=ff_conf)
        mlp_conf = MLPConfig().mlp
        self.mlp = mlp_conf
        self.ff = {'num_node_types': 1}


class FeedForward(nn.Module):
    def __init__(self, conf):
        super(FeedForward, self).__init__()
        mlp_conf = conf.mlp
        num_node_types = conf.ff['num_node_types']

        node_updater_dict = {}
        for i in range(num_node_types):
            node_updater_dict['node_updater{}'.format(i)] = MultiLayerPerceptron(**mlp_conf)
        self.node_updater = nn.ModuleDict(node_updater_dict)

    def forward(self, graph, node_feature, update_node_type_indices=[NODE_ALLY]):
        graph.ndata['node_feature'] = node_feature
        for ntype_idx in update_node_type_indices:
            node_index = get_filtered_node_index_by_type(graph, ntype_idx)
            apply_func = partial(self.apply_node_function, ntype_idx=ntype_idx)
            graph.apply_nodes(func=apply_func, v=node_index)

        _ = graph.ndata.pop('node_feature')
        updated_node_feature = graph.ndata.pop('updated_node_feature')
        return updated_node_feature

    def apply_node_function(self, nodes, ntype_idx):
        updater = self.node_updater['node_updater{}'.format(ntype_idx)]
        return {'updated_node_feature': updater(nodes.data['node_feature'])}
