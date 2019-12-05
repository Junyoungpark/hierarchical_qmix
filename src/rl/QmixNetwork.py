import dgl
import torch

from src.nn.MLP import MLPConfig, MultiLayerPerceptron as MLP
from src.nn.GraphConvolution import GraphConvolutionLayer
from src.rl.Qmixer import QMixer, QMixerConfig
from src.config.ConfigBase import ConfigBase
from src.util.graph_util import get_number_of_ally_nodes


class QmixNetworkConfig(ConfigBase):
    def __init__(self, submixer_conf=None, supmixer_conf=None):
        super(QmixNetworkConfig, self).__init__(submixer=submixer_conf, supmixer=supmixer_conf)
        self.submixer = QMixerConfig()
        self.supmixer = {'prefix': 'supmixer'}.update(MLPConfig().mlp)


class QmixNetwork(torch.nn.Module):
    def __init__(self, conf):
        super(QmixNetwork, self).__init__()

        self.submixer_conf = conf.submixer()
        self.supmixer_gc_conf = conf.supmixer_gc()
        self.supmixer_mlp_conf = conf.supmixer_mlp()

        self.submixer = QMixer(self.submixer_conf)
        self.supmixer = GraphConvolutionLayer(self.supmixer_gc_conf)
        self.supmixer_b = MLP(self.supmixer_mlp_conf)

    def forward(self, graph, node_feature, qs):
        sub_q_ret_dict = self.submixer(graph, node_feature, qs)

        aggregated_feat = sub_q_ret_dict['feat']  # [#. graph x num_cluster x feature_dim]
        aggregated_q = sub_q_ret_dict['qs']  # [#. graph x #.cluster]
        ws = sub_q_ret_dict['ws']  # [#. allies x #. clusters]

        #### slow implementation ####
        graphs = dgl.unbatch(graph)
        nums_ally = get_number_of_ally_nodes(graphs)

        adj_mats = []
        for w in ws.split(nums_ally):
            adj_mat = w.t().mm(w)  # [#. clusters x #. clusters]
            adj_mats.append(adj_mat)
        adj_mats = torch.stack(adj_mats)  # [#. graph x #. clusters x #. clusters]
        #### slow implementation ####

        sup_ws = self.supmixer(input=aggregated_feat, adj=adj_mats)  # [#. graph x #. clusters x 1]

        sup_weighted_qs = sup_ws * aggregated_q.unsqueeze(dim=-1)  # [#. graph x #.cluster x 1]
        sup_qs = sup_weighted_qs.sum(dim=1)

        return sup_qs
