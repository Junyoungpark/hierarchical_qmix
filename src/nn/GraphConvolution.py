import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolutionLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    @staticmethod
    def Laplacian_norm(mx):
        """
        This implementation ignores the self-connection of adj matrix!
        """

        assert mx.dim() == 3, "assume 'mx' is a batched matrix"
        colsum = mx.sum(dim=1)  # [# input matrix, # out_feat_dim]
        d_inv_sqrt = torch.pow(colsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # [# input matrix x # out_feat_dim]
        d_inv_sqrt_mat = torch.diag_embed(d_inv_sqrt)  # [# input matrix x # out_feat_dim x # out_feat_dim]
        normed_mx = d_inv_sqrt_mat.bmm(mx).bmm(d_inv_sqrt_mat)
        return normed_mx

    def forward(self, input, adj, adj_normed=False):

        if not adj_normed:
            adj = self.Laplacian_norm(adj)

        support = input.matmul(self.weight)  # [# graphs x # nodes per graph x # out_feat_dim]
        output = support.bmm(adj)

        if self.bias is not None:
            output = output + self.bias

        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


if __name__ == "__main__":
    batch_size = 5
    num_nodes = 3
    in_feats = 2
    out_feats = 1

    gc = GraphConvolutionLayer(in_feats, out_feats)

    adj = torch.rand(size=(batch_size, out_feats, out_feats))
    input = torch.rand(size=(batch_size, num_nodes, in_feats))
    output = gc(input, adj)

    print(output)