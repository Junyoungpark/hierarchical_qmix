import dgl
import torch

from src.nn.RelationalGraphNetwork import RelationalGraphNetwork, RelationalGraphNetworkConfig
from src.nn.RNNEncoder import RNNEncoder
from src.config.ConfigBase import ConfigBase


class MultiStepInputGraphNetworkConfig(ConfigBase):

    def __init__(self,
                 name='multistepgnn',
                 hist_rnn_conf=None,
                 hist_enc_conf=None,
                 curr_enc_conf=None
                 ):
        super(MultiStepInputGraphNetworkConfig, self).__init__(name=name, hist_rnn=hist_rnn_conf,
                                                               hist_enc=hist_enc_conf, curr_enc=curr_enc_conf)

        gnn_conf = RelationalGraphNetworkConfig().gnn

        self.hist_rnn = {
            'rnn_type': 'GRU',
            'input_size': 19,
            'hidden_size': 32,
            'num_layers': 2,
            'batch_first': True}

        self.hist_enc = gnn_conf
        self.curr_enc = gnn_conf


class MultiStepInputGraphNetwork(torch.nn.Module):

    def __init__(self, conf):
        super(MultiStepInputGraphNetwork, self).__init__()
        self.conf = conf

        rnn_conf = conf.hist_rnn
        rnn_type = rnn_conf.pop('rnn_type')
        rnn = getattr(torch.nn, rnn_type)
        self.hist_rnn = rnn(**rnn_conf)

        hist_enc_conf = conf.hist_enc
        self.hist_one_step_enc = RelationalGraphNetwork(**hist_enc_conf)

        self.hist_encoder = RNNEncoder(rnn=self.hist_rnn, one_step_encoder=self.hist_one_step_enc)

        curr_enc_conf = conf.curr_enc
        self.curr_encoder = RelationalGraphNetwork(**curr_enc_conf)
        self.out_dim = curr_enc_conf['output_node_dim'] + rnn_conf['hidden_size']

    def forward(self, num_time_steps, hist_graph, hist_feature,
                curr_graph, curr_feature):

        h_enc_out, h_enc_hidden = self.hist_encoder(num_time_steps, hist_graph, hist_feature)
        device = h_enc_out.device

        # recent_hist_enc : slice of the last RNN layer's hidden
        recent_h_enc = h_enc_out[:, -1, :]  # [Batch size x rnn hidden]
        c_enc_out = self.curr_encoder(curr_graph, curr_feature)

        if isinstance(curr_graph, dgl.BatchedDGLGraph):
            c_units = curr_graph.batch_num_nodes
            c_units = torch.tensor(c_units, dtype=torch.long, device=device)
        else:
            c_units = curr_graph.number_of_nodes()
        recent_h_enc = recent_h_enc.repeat_interleave(c_units, dim=0)
        c_encoded_node_feature = torch.cat([recent_h_enc, c_enc_out], dim=1)
        return c_encoded_node_feature
