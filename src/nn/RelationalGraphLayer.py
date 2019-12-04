import torch
import torch.nn as nn


class RelationalGraphLayer(torch.nn.Module):
    """
    Relational Graph Network layer
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 node_types: list,
                 edge_types: list,
                 updater_conf: dict,
                 use_residual: bool = False,
                 use_concat: bool = False):
        """
        :param input_dim:
        :param output_dim:
        :param node_types:
        :param edge_types:
        """

        super(RelationalGraphLayer, self).__init__()

        # infer inter-layer hook type
        