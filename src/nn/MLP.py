import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.nn.activations import get_nn_activation


class NoisyLinear(nn.Linear):
    # Adapted from the original source
    # https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py

    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.sigma_weight, self.sigma_init)
        torch.nn.init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, x):
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight,
                        self.bias + self.sigma_bias * self.epsilon_bias)

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class LinearModule(nn.Module):
    def __init__(self,
                 activation: 'str',
                 norm: 'str' = None,
                 dropout_p: 'float' = 0.0,
                 weight_init: 'str' = 'xavier',
                 use_noisy: bool = False,
                 **linear_kwargs):
        super(LinearModule, self).__init__()

        # layers
        if use_noisy:
            linear_layer = NoisyLinear(**linear_kwargs)
        else:
            linear_layer = torch.nn.Linear(**linear_kwargs)

        self.linear_layer = linear_layer
        self.dropout_layer = torch.nn.Dropout(dropout_p)
        self.activation_layer = get_nn_activation(activation)

        self.weight_init = weight_init
        self.activation = activation
        self.norm = norm

        # apply weight initialization methods
        self.apply_weight_init(self.linear_layer, self.weight_init)

        if norm == 'batch':
            self.norm_layer = torch.nn.BatchNorm1d(self.linear_layer.out_features)
        elif norm == 'layer':
            self.norm_layer = torch.nn.LayerNorm(self.linear_layer.out_features)
        elif norm == 'spectral':
            self.linear_layer = torch.nn.utils.spectral_norm(self.linear_layer)
            self.norm_layer = torch.nn.Identity()
        elif norm is None:
            self.norm_layer = torch.nn.Identity()
        else:
            raise RuntimeError("Not implemented normalization function!")

    def apply_weight_init(self, tensor, weight_init=None):
        if weight_init is None:
            pass  # do not apply weight init
        elif weight_init == "normal":
            torch.nn.init.normal_(tensor.weight, std=0.3)
            torch.nn.init.constant_(tensor.bias, 0.0)
        elif weight_init == "kaiming_normal":
            if self.activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
                torch.nn.init.kaiming_normal_(tensor.weight, nonlinearity=self.activation)
                torch.nn.init.constant_(tensor.bias, 0.0)
            else:
                pass
        elif weight_init == "xavier":
            torch.nn.init.xavier_uniform_(tensor.weight)
        else:
            raise NotImplementedError("MLP initializer {} is not supported".format(weight_init))

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        x = self.dropout_layer(x)
        return x


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 num_neurons,
                 activation,
                 out_activation,
                 normalization=None,
                 weight_init='xavier',
                 dropout_probability=0.0):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = input_dimension
        self.output_dim = output_dimension
        self.num_neurons = num_neurons

        _list_norm = self.check_input_spec(normalization)
        _input_norm = True if _list_norm and len(normalization) == 1 else False
        _list_act = self.check_input_spec(activation)
        _list_drop_p = self.check_input_spec(dropout_probability)

        input_dims = [input_dimension] + num_neurons
        output_dims = num_neurons + [output_dimension]

        # Input -> the last hidden layer
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims[:-1], output_dims[:-1])):
            norm = normalization[i] if _list_norm else normalization
            norm = None if _input_norm and i != 0 else norm
            act = activation[i] if _list_act else activation
            drop_p = dropout_probability[i] if _list_drop_p else dropout_probability

            linear_module = LinearModule(in_features=in_dim, out_features=out_dim,
                                         activation=act, norm=norm, dropout_p=drop_p, weight_init=weight_init)
            self.layers.append(linear_module)

        output_layer = LinearModule(in_features=input_dims[-1], out_features=output_dims[-1],
                                    activation=out_activation, norm=None, dropout_p=0.0, weight_init=weight_init)
        self.layers.append(output_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def check_input_spec(self, input_spec):
        if isinstance(input_spec, list):
            # output layer will not be normalized
            assert len(input_spec) == len(self.num_neurons) + 1, "the length of input_spec list should " \
                                                                 "match with the number of hidden layers + 1"
            _list_type = True
        else:
            _list_type = False

        return _list_type
