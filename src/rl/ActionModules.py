from functools import partial
import torch

from src.nn.MLP import MultiLayerPerceptron as MLP
from src.util.graph_util import get_filtered_edge_index_by_type

from src.config.nn_config import VERY_LARGE_NUMBER


class MoveModule(torch.nn.Module):

    def __init__(self,
                 mlp_config: dict,
                 move_dim: int = 4):
        super(MoveModule, self).__init__()

        mlp_config['output_dimension'] = move_dim
        self.move_argument_calculator = MLP(**mlp_config)

    def forward(self, graph, node_feature):
        graph.ndata['node_feature'] = node_feature
        graph.apply_nodes(func=self.apply_node_function)
        move_argument = graph.ndata.pop('move_argument')
        return move_argument

    def apply_node_function(self, nodes):
        input_node_feature = nodes.data['node_feature']
        move_argument = self.move_argument_calculator(input_node_feature)
        return {'move_argument': move_argument}


class AttackModule(torch.nn.Module):

    def __init__(self,
                 mlp_config: dict):
        super(AttackModule, self).__init__()
        mlp_config['input_dimension'] = mlp_config['input_dimension'] * 2
        mlp_config['output_dimension'] = 1
        self.attack_argument_calculator = MLP(**mlp_config)

    def message_function(self, edges):
        enemy_node_features = edges.src['node_feature']  # Enemy units' feature
        enemy_tag = edges.src['tag']
        ally_node_features = edges.dst['node_feature']  # Ally units' feature
        attack_argument_input = torch.cat((ally_node_features, enemy_node_features), dim=-1)
        attack_argument = self.attack_argument_calculator(attack_argument_input)
        return {'attack_argument': attack_argument, 'enemy_tag': enemy_tag}

    @staticmethod
    def get_action_reduce_function(nodes, num_enemy_units):
        mailbox_attack_argument = nodes.mailbox['attack_argument']
        device = mailbox_attack_argument.device

        attack_argument = torch.ones(size=(len(nodes), num_enemy_units), device=device) * - VERY_LARGE_NUMBER
        attack_argument[:, :mailbox_attack_argument.shape[1]] = mailbox_attack_argument.squeeze(dim=-1)

        mailbox_enemy_tag = nodes.mailbox['enemy_tag']
        enemy_tag = torch.ones(size=(len(nodes), num_enemy_units), dtype=torch.long, device=device)
        enemy_tag[:, :mailbox_enemy_tag.shape[1]] = mailbox_enemy_tag
        return {'attack_argument': attack_argument, 'enemy_tag': enemy_tag}

    def forward(self, graph, node_feature, maximum_num_enemy: int, attack_edge_type_index: int):
        num_total_nodes = graph.number_of_nodes()
        graph.ndata['node_feature'] = node_feature
        edge_index = get_filtered_edge_index_by_type(graph, attack_edge_type_index)
        reduce_func = partial(self.get_action_reduce_function, num_enemy_units=maximum_num_enemy)
        graph.send_and_recv(edges=edge_index,
                            message_func=self.message_function,
                            reduce_func=reduce_func)
        if len(edge_index) != 0:
            attack_argument = graph.ndata.pop('attack_argument')
        else:
            attack_argument = torch.ones(size=(num_total_nodes, maximum_num_enemy)) * - VERY_LARGE_NUMBER
        return attack_argument


class HoldModule(torch.nn.Module):
    def __init__(self,
                 mlp_config):
        super(HoldModule, self).__init__()
        self.hold_argument_calculator = MLP(**mlp_config)

    def forward(self, graph, node_feature):
        graph.ndata['node_feature'] = node_feature
        graph.apply_nodes(func=self.apply_function)
        return graph.ndata.pop('hold_argument')

    def apply_function(self, nodes):
        node_features = nodes.data['node_feature']
        hold_argument = self.hold_argument_calculator(node_features)
        return {'hold_argument': hold_argument}
