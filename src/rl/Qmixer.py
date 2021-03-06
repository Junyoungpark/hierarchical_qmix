import dgl
import torch
import torch.nn as nn

from src.nn.MLP import MultiLayerPerceptron as MLP, MLPConfig
from src.nn.RelationalGraphNetwork import RelationalGraphNetwork
from src.nn.RelationalGraphNetwork import RelationalGraphNetworkConfig as RGNConfig

from src.config.ConfigBase import ConfigBase

from src.util.graph_util import get_filtered_node_index_by_type, get_number_of_ally_nodes
from src.config.graph_config import NODE_ALLY, NODE_ENEMY, EDGE_ALLY, EDGE_ALLY_TO_ENEMY, EDGE_ENEMY
from src.config.nn_config import VERY_SMALL_NUMBER


class QmixerConfig(ConfigBase):

    def __init__(self, name='qmixer', mixer_conf=None, b_net_conf=None, w_net_conf=None):
        super(QmixerConfig, self).__init__(name=name, mixer=mixer_conf, b_net=b_net_conf, w_net=w_net_conf)

        self.mixer = {'num_clusters': 4, 'use_clipped_score': True}
        self.b_net = MLPConfig().mlp
        self.b_net['input_dimension'] = 51
        self.b_net['output_dimension'] = self.mixer['num_clusters']
        self.b_net['out_activation'] = None

        self.w_net = RGNConfig().gnn
        self.w_net['input_node_dim'] = 51
        self.w_net['output_node_dim'] = self.mixer['num_clusters']
        self.w_net['num_hidden_layers'] = 0
        self.w_net['node_types'] = [NODE_ALLY, NODE_ENEMY]
        self.w_net['edge_types'] = [EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY]


class Qmixer(nn.Module):

    def __init__(self, conf):
        super(Qmixer, self).__init__()

        self.num_clusters = conf.mixer['num_clusters']

        b_net_conf = conf.b_net
        w_net_conf = conf.w_net

        self.q_b_net = MLP(**b_net_conf)
        self.w_net = RelationalGraphNetwork(**w_net_conf)
        self.use_clipped_score = conf.mixer['use_clipped_score']

    def get_w(self, graph, node_feature):
        ws = self.w_net(graph, node_feature)  # [#. allies x #. clusters]
        ally_indices = get_filtered_node_index_by_type(graph, NODE_ALLY)
        ally_ws = ws[ally_indices, :]  # [#. allies x #. clusters]
        if self.use_clipped_score:
            ally_ws = ally_ws.clamp(min=VERY_SMALL_NUMBER, max=10)

        ally_ws = torch.nn.functional.softmax(ally_ws, dim=1)
        return ally_ws

    def get_feat(self, graph, node_feature, ws=None):

        if ws is None:
            ws = self.get_w(graph, node_feature)

        if isinstance(graph, dgl.BatchedDGLGraph):
            single_graph = False
        else:
            single_graph = True

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

        if single_graph:
            weighted_feat = weighted_feat.unsqueeze(0)

        wf = weighted_feat.transpose(2, 1)  # [#. graph x num_cluster x feature_dim]
        _nf = node_feature.unsqueeze(-1)  # [# nodes x # features x 1]

        # compute group-wise compatibility scores

        if single_graph:
            num_ally_nodes = get_number_of_ally_nodes([graph])
        else:
            num_ally_nodes = get_number_of_ally_nodes(dgl.unbatch(graph))

        repeat_wf = torch.repeat_interleave(wf,
                                            torch.tensor(num_ally_nodes, device=node_feature.device),
                                            dim=0)  # [# allies x # clusters x feature_dim]

        ally_nf_expanded = node_feature[ally_indices, :].unsqueeze(-1)  # [# allies x feature_dim x 1]

        group_dot_prd = (ally_nf_expanded * repeat_wf.transpose(2, 1)).sum(1)  # [ # allies x # clusters ]

        nf_norm_allies = torch.norm(_nf[ally_indices, :], dim=1)  # [# allies x 1]
        wf_norm = torch.norm(repeat_wf, dim=2)  # [# allies x # clusters ]

        ally_normed_group_dot_prd = group_dot_prd / (nf_norm_allies * wf_norm)  # [# allies x # clusters]
        normed_group_dot_prd = torch.zeros(size=(node_feature.shape[0], ally_normed_group_dot_prd.shape[1]),
                                           device=node_feature.device)
        normed_group_dot_prd[ally_indices, :] = ally_normed_group_dot_prd

        _ = graph.ndata.pop('weighted_feat')

        return wf, ally_normed_group_dot_prd, normed_group_dot_prd

    def get_q(self, graph, node_feature, qs, ws=None):
        device = node_feature.device
        ally_indices = get_filtered_node_index_by_type(graph, NODE_ALLY)

        # compute weighted sum of qs
        if ws is None:
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
        aggregated_feat, ally_normed_group_dot_prd, normed_group_dot_prd = self.get_feat(graph, node_feature, ws)
        aggregated_q = self.get_q(graph, node_feature, qs)

        ret_dict = dict()
        ret_dict['qs'] = aggregated_q
        ret_dict['ws'] = ws
        ret_dict['feat'] = aggregated_feat
        ret_dict['ally_normed_group_dot_prd'] = ally_normed_group_dot_prd
        ret_dict['normed_group_dot_prd'] = normed_group_dot_prd
        return ret_dict


if __name__ == "__main__":
    conf = QmixerConfig()
    # conf = QMixerConfig()
    # QMixer(conf)
