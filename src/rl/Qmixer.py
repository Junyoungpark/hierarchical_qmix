import dgl
import torch
import torch.nn as nn

from src.nn.MLP import MultiLayerPerceptron as MLP, MLPConfig
from src.nn.RelationalGraphNetwork import RelationalGraphNetwork
from src.nn.RelationalGraphNetwork import RelationalGraphNetworkConfig as RGNConfig

from src.config.graph_config import NODE_ALLY
from src.config.ConfigBase import ConfigBase

from src.util.graph_util import get_filtered_node_index_by_type


class QMixerConfig(ConfigBase):

    def __init__(self, mixer_conf, b_net_conf, w_net_conf):
        super(QMixerConfig, self).__init__(mixer=mixer_conf,
                                           b_net=b_net_conf,
                                           w_net=w_net_conf)

        self.mixer = {
            'prefix': 'mixer',
            'num_clusters': 3,
        }

        self.b_net = {'prefix': 'mixer-b-net'}.update(MLPConfig().mlp)
        self.w_net = {'prefix': 'mixer-w-net'}.update(RGNConfig().gnn)


class QMixer(nn.Module):

    def __init__(self, conf):
        super(QMixer, self).__init__()

        self.conf = conf
        self.num_clusters = conf.mixer['num_clusters']

        b_net_conf = conf.b_net()
        b_net_conf['output_dimension'] = self.num_clusters

        w_net_conf = conf.w_net()
        w_net_conf['output_node_dim'] = self.num_clusters

        self.q_b_net = MLP(**b_net_conf)
        self.w_net = RelationalGraphNetwork(**w_net_conf)

    def get_w(self, graph, node_feature):
        ws = self.w_net(graph, node_feature)  # [#. allies x #. clusters]
        ally_indices = get_filtered_node_index_by_type(graph, NODE_ALLY)
        ally_ws = ws[ally_indices, :]  # [#. allies x #. clusters]
        return ally_ws

    def get_feat(self, graph, node_feature):
        ws = self.get_w(graph, node_feature)
        ally_indices = get_filtered_node_index_by_type(graph, NODE_ALLY)
        ally_node_feature = node_feature[ally_indices, :]  # [#. allies x feature dim]

        num_allies, feat_dim = ally_node_feature.shape[0], ally_node_feature.shape[1]

        ally_node_feature = ally_node_feature.repeat(1, 1, self.num_clusters)  # [#. allies x feature dim x #. clusters]

        # [#. allies x feature dim x #. cluster]
        weighted_feat = ws.view(num_allies,
                                feat_dim,
                                self.num_clusters) * ally_node_feature.unsqueeze(dim=-1)

        graph.ndata['weighted_feat'] = weighted_feat.view(num_allies, -1)
        weighted_feat = dgl.sum_nodes(graph, 'weighted_feat')  # [#. graph x feature dim x #. clusters]
        _ = graph.ndata.pop('weighted_feat')

        if isinstance(graph, dgl.BatchedDGLGraph):
            num_graphs = graph.batch_size
        else:
            num_graphs = 1

        weighted_feat = weighted_feat.reshape(num_graphs, feat_dim, self.num_clusters)
        return weighted_feat.transpose(0, 2, 1)  # [#. graph x num_cluster x feature_dim]

    def get_q(self, graph, node_feature, qs):
        device = node_feature.device
        ally_indices = get_filtered_node_index_by_type(graph, NODE_ALLY)

        # compute weighted sum of qs
        ws = self.get_w(graph, node_feature)  # [#. allies x #. clusters]
        weighted_q = qs.view(-1, 1) * ws  # [#. allies x #. clusters]

        qs = torch.zeros(size=(graph.number_of_nodes(), self.num_clusters), device=device)
        qs[ally_indices, :] = weighted_q

        graph.ndata['q'] = qs
        q_aggregated = dgl.sum_nodes(graph, 'q')  # [#. graph x #. clusters]

        # compute state_dependent_bias
        graph.ndata['node_feature'] = node_feature
        sum_node_feature = dgl.sum_nodes(graph, 'node_feature')  # [#. graph x feature dim]
        q_v = self.q_b_net(sum_node_feature)  # [#. graph x #. clusters]

        _ = graph.ndata.pop('node_feature')
        _ = graph.ndata.pop('q')
        q_aggregated = q_aggregated + q_v

        return q_aggregated  # [#. graph x #. clusters]

    def forward(self, graph, node_feature, qs):
        ws = self.get_w(graph, node_feature)
        aggregated_feat = self.get_feat(graph, node_feature)
        aggregated_q = self.get_q(graph, node_feature, qs)

        ret_dict = dict()
        ret_dict['qs'] = aggregated_q
        ret_dict['ws'] = ws
        ret_dict['feat'] = aggregated_feat
        return ret_dict
