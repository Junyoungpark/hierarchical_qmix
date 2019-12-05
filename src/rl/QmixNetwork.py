import dgl
import torch

from src.nn.MLP import MLPConfig, MultiLayerPerceptron as MLP
from src.nn.GraphConvolution import GraphConvolutionLayer
from src.rl.Qmixer import Qmixer, QmixerConfig
from src.config.ConfigBase import ConfigBase
from src.util.graph_util import get_number_of_ally_nodes


class QmixNetworkConfig(ConfigBase):
    def __init__(self, submixer_conf=None, supmixer_gc_conf=None, supmixer_mlp_conf=None):
        super(QmixNetworkConfig, self).__init__(submixer=submixer_conf, supmixer_gc=supmixer_gc_conf,
                                                supmixer_mlp=supmixer_mlp_conf)
        self.submixer = QmixerConfig()
        self.supmixer_mlp = {'prefix': 'supmixer_mlp', **MLPConfig().mlp}
        self.supmixer_gc = {'prefix': 'supmixer_gc',
                            'in_features': 3,
                            'out_features': 1,
                            'bias': True}


class QmixNetwork(torch.nn.Module):
    def __init__(self, conf):
        super(QmixNetwork, self).__init__()

        self.submixer_conf = conf.submixer
        self.supmixer_gc_conf = conf.supmixer_gc
        self.supmixer_mlp_conf = conf.supmixer_mlp

        self.submixer = Qmixer(self.submixer_conf)
        self.supmixer = GraphConvolutionLayer(**self.supmixer_gc_conf)
        self.supmixer_b = MLP(**self.supmixer_mlp_conf)

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

        if isinstance(graph, dgl.BatchedDGLGraph):
            num_graphs = graph.batch_size
        else:
            num_graphs = 1

        sup_q_bs = self.supmixer_b((aggregated_feat.view(num_graphs, -1)))  # [#. graph x  1]
        sup_qs = sup_qs + sup_q_bs

        return sup_qs


if __name__ == "__main__":
    conf = QmixNetworkConfig()
    QmixNetwork(conf)
