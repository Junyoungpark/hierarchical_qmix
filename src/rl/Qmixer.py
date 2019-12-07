import dgl
import torch
import torch.nn as nn

from src.nn.MLP import MultiLayerPerceptron as MLP, MLPConfig
from src.nn.RelationalGraphNetwork import RelationalGraphNetwork
from src.nn.RelationalGraphNetwork import RelationalGraphNetworkConfig as RGNConfig

from src.config.graph_config import NODE_ALLY
from src.config.ConfigBase import ConfigBase

from src.util.graph_util import get_filtered_node_index_by_type
from src.util.train_util import dn

class QmixerConfig(ConfigBase):

    def __init__(self, name='qmixer', mixer_conf=None, b_net_conf=None, w_net_conf=None):
        super(QmixerConfig, self).__init__(name=name, mixer=mixer_conf, b_net=b_net_conf, w_net=w_net_conf)

        self.mixer = {'num_clusters': 3}
        self.b_net = MLPConfig().mlp
        self.b_net['input_dimension'] = 19
        self.b_net['output_dimension'] = self.mixer['num_clusters']

        self.w_net = RGNConfig().gnn
        self.w_net['output_node_dim'] = self.mixer['num_clusters']


class Qmixer(nn.Module):

    def __init__(self, conf):
        super(Qmixer, self).__init__()

        self.conf = conf
        self.num_clusters = conf.mixer['num_clusters']

        b_net_conf = conf.b_net
        w_net_conf = conf.w_net

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

        ally_node_feature = ally_node_feature.unsqueeze(dim=-1)
        ally_node_feature = ally_node_feature.repeat(1, 1, self.num_clusters)  # [#. allies x feature dim x #. clusters]

        # [#. allies x feature dim x #. cluster]
        weighted_feat = ws.view(num_allies, 1, self.num_clusters) * ally_node_feature

        _wf = torch.zeros(size=(graph.number_of_nodes(), feat_dim, self.num_clusters), device=ws.device)
        _wf[ally_indices, :] = weighted_feat

        graph.ndata['weighted_feat'] = _wf
        weighted_feat = dgl.sum_nodes(graph, 'weighted_feat')  # [#. graph x feature dim x #. clusters]
        _ = graph.ndata.pop('weighted_feat')

        return weighted_feat.transpose(2, 1)  # [#. graph x num_cluster x feature_dim]

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


if __name__ == "__main__":
    conf = QmixerConfig()
    # conf = QMixerConfig()
    # QMixer(conf)
