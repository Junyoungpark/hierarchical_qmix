import torch

from src.nn.RelationalGraphNetwork import RelationalGraphNetworkConfig, RelationalGraphNetwork
from src.nn.FeedForward import FeedForwardConfig, FeedForward
from src.config.ConfigBase import ConfigBase
from src.util.graph_util import NODE_ALLY

from src.util.train_util import dn


class QmixNetwork2Config(ConfigBase):
    def __init__(self, name='qmixnetwork2', gnn_conf=None, ff_conf=None):
        super(QmixNetwork2Config, self).__init__(name=name, gnn=gnn_conf, ff=ff_conf)

        self.gnn = RelationalGraphNetworkConfig().gnn
        self.ff = FeedForwardConfig()


class QmixNetwork2(torch.nn.Module):
    def __init__(self, conf):
        super(QmixNetwork2, self).__init__()
        gnn_conf = conf.gnn_conf
        ff_conf = conf.ff_conf

        self.w_gn = RelationalGraphNetwork(**gnn_conf)
        self.w_ff = FeedForward(**ff_conf)
        self.v_gn = RelationalGraphNetwork(**gnn_conf)
        self.v_ff = FeedForward(**ff_conf)

    def forward(self, graph, node_feature, qs, ally_node_type_index=NODE_ALLY):
        pass
