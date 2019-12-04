import dgl
import torch


class RNNEncoder(torch.nn.Module):

    def __init__(self, rnn, one_step_encoder):
        super(RNNEncoder, self).__init__()
        self.rnn = rnn
        self.one_step_encoder = one_step_encoder

    def forward(self,
                num_time_steps: int,
                batch_time_batched_graph: dgl.BatchedDGLGraph,
                node_feat):
        """
        :param num_time_steps: (int) number of graphs per input trajectory
        :param batch_time_batched_graph: (dgl.BatchedDGLGraph) The order of batching is expected as follows

        * g_(i,j) = i th trajectories' j th graph
        * nt = num_time_steps
        * bs = batch_size

        the expect order of batching is:
        [g_(0,0), g_(0,1), ... g_(0,nt), g_(1,0), g_(1,1), ..., g_(1,nt), g_(2,0), ..., g_(bs, 0), ... g_(bs, nt)]

        :param node_feat:
        :return:
        """
        assert batch_time_batched_graph.batch_size % num_time_steps == 0, "Input batch graph has the unexpected size"
        embedded_feature_dict = self.one_step_encoder(batch_time_batched_graph, node_feat)
        batch_time_batched_graph.ndata['node_feature'] = embedded_feature_dict
        readouts = dgl.sum_nodes(batch_time_batched_graph, 'node_feature')
        readouts = readouts.view(-1, num_time_steps, readouts.shape[-1])
        rnn_out, rnn_hidden = self.rnn(readouts)
        return rnn_out, rnn_hidden
