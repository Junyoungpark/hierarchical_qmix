import dgl
import torch

from src.nn.MLP import MLPConfig, MultiLayerPerceptron as MLP
from src.nn.GCN import GCN, GCNConfig
from src.rl.Qmixer import Qmixer, QmixerConfig
from src.config.ConfigBase import ConfigBase
from src.util.graph_util import get_number_of_ally_nodes
from src.util.train_util import dn


class QmixNetworkConfig(ConfigBase):
    def __init__(self, name='qmixnetwork', submixer_conf=None, supmixer_gc_conf=None, supmixer_mlp_conf=None,
                 supmixer_gnn_conf=None):
        super(QmixNetworkConfig, self).__init__(name=name, submixer=submixer_conf, supmixer_gc=supmixer_gc_conf,
                                                supmixer_mlp=supmixer_mlp_conf, supermixer_gnn=supmixer_gnn_conf)
        self.submixer = QmixerConfig()
        self.supmixer_gc = GCNConfig().gcn

        nf_dim = 51

        self.supmixer_gc['in_features'] = nf_dim
        self.supmixer_mlp = MLPConfig().mlp
        self.supmixer_mlp['input_dimension'] = nf_dim
        self.supmixer_mlp['output_dimension'] = 1
        self.supmixer_mlp['out_activation'] = None


class QmixNetwork(torch.nn.Module):
    def __init__(self, conf):
        super(QmixNetwork, self).__init__()

        self.submixer_conf = conf.submixer
        self.supmixer_gc_conf = conf.supmixer_gc
        self.supmixer_mlp_conf = conf.supmixer_mlp

        self.submixer = Qmixer(self.submixer_conf)

        # choose among two options on supmixer

        self.supmixer = GCN(**self.supmixer_gc_conf) # opt 1: GCN supmixer
        # self.supmixer = MLP(**self.supmixer_mlp_conf) # opt 2: MLP supmixer
        self.supmixer_b = MLP(**self.supmixer_mlp_conf)

        self.n_clusters = self.submixer_conf.mixer['num_clusters']

    def forward(self, graph, node_feature, qs):
        sub_q_ret_dict = self.submixer(graph, node_feature, qs)

        aggregated_feat = sub_q_ret_dict['feat']  # [#. graph x num_cluster x feature_dim]
        aggregated_q = sub_q_ret_dict['qs']  # [#. graph x #.cluster]
        ws = sub_q_ret_dict['ws']  # [#. allies x #. clusters]

        #### GCN: slow implementation ####
        graphs = dgl.unbatch(graph)
        nums_ally = get_number_of_ally_nodes(graphs)

        adj_mats = []
        for w in ws.split(nums_ally):
            adj_mat = w.t().mm(w) + torch.eye(self.n_clusters, device=ws.device)  # [#. clusters x #. clusters]
            adj_mats.append(adj_mat)
        adj_mats = torch.stack(adj_mats)  # [#. graph x #. clusters x #. clusters]
        sup_ws = self.supmixer(input=aggregated_feat, adj=adj_mats)  # [#. graph x #. clusters x 1]
        #### slow implementation ####

        #### MLP Style ####
        # graphs = dgl.unbatch(graph)
        # nums_ally = get_number_of_ally_nodes(graphs)
        #
        # wss = []
        # for w in ws.split(nums_ally):
        #     w_agg = w.sum(0)
        #     wss.append(w_agg)
        # wss = torch.stack(wss)  # [# graph X # clusters]
        # aggregated_feat = aggregated_feat * wss.unsqueeze(dim=-1)  # [# graph X # clusters X feature dim]
        #
        # sup_ws = self.supmixer(aggregated_feat)
        #### MLP Style ####

        sup_ws = torch.nn.functional.softmax(sup_ws, dim=1)

        sup_weighted_qs = sup_ws * aggregated_q.unsqueeze(dim=-1)  # [#. graph x #.cluster x 1]
        sup_qs = sup_weighted_qs.sum(dim=1)

        sup_q_bs = self.supmixer_b((aggregated_feat.sum(dim=1)))  # [#. graph x  1]
        sup_qs = sup_qs + sup_q_bs

        return sup_qs.view(-1)


if __name__ == "__main__":
    conf = QmixNetworkConfig()
    QmixNetwork(conf)
