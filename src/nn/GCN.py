import torch.nn as nn
from src.nn.activations import get_nn_activation
from src.nn.GraphConvolution import GraphConvolutionLayer
from src.config.ConfigBase import ConfigBase


class GCNConfig(ConfigBase):

    def __init__(self, name='gcn', gcn_conf=None):
        super(GCNConfig, self).__init__(name=name, gcn=gcn_conf)
        self.gcn = {
            'in_features': 51,
            'hidden_features': 64,
            'out_features': 1,
            'num_hidden_layers': 2,
            'hidden_act': 'mish'
        }


class GCN(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 num_hidden_layers,
                 hidden_act):
        super(GCN, self).__init__()

        input_dims = [in_features] + num_hidden_layers * [hidden_features]
        output_dims = num_hidden_layers * [hidden_features] + [out_features]

        self.layers = nn.ModuleList()
        for input_dim, output_dim in zip(input_dims, output_dims):
            layer = GraphConvolutionLayer(in_features=input_dim,
                                          out_features=output_dim)

            self.layers.append(layer)

        self.act = get_nn_activation(hidden_act)

    def forward(self, input, adj):

        normed_adj = self.layers[0].Laplacian_norm(adj)
        x = input
        for layer in self.layers:
            x = layer(input=x, adj=normed_adj, adj_normed=True)
            x = self.act(x)
        return x
