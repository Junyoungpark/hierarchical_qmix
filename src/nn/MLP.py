import torch.nn as nn
from src.nn.Linear import LinearModule
from src.config.ConfigBase import ConfigBase


class MLPConfig(ConfigBase):

    def __init__(self, mlp_conf=None):
        super(MLPConfig, self).__init__(mlp=mlp_conf)

        self.mlp = {
            'prefix': 'mlp',
            'input_dimension': 32,
            'output_dimension': 32,
            'activation': 'mish',
            'out_activation': None,
            'num_neurons': [64, 64],
            'normalization': None,
            'weight_init': 'xavier',
            'dropout_probability': 0.0,
            'use_noisy': False}


class rMLPConfig(ConfigBase):

    def __init__(self, name='mlp', mlp_conf=None):
        super(rMLPConfig, self).__init__(name=name, mlp=mlp_conf)

        self.mlp = {
            'prefix': 'mlp',
            'input_dimension': 32,
            'output_dimension': 32,
            'activation': 'mish',
            'out_activation': None,
            'num_neurons': [64, 64],
            'normalization': None,
            'weight_init': 'xavier',
            'dropout_probability': 0.0,
            'use_noisy': False}


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 num_neurons,
                 activation,
                 out_activation,
                 normalization=None,
                 weight_init='xavier',
                 dropout_probability=0.0,
                 use_noisy=False):

        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = input_dimension
        self.output_dim = output_dimension
        self.num_neurons = num_neurons
        self.use_noisy = use_noisy

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

            linear_module = LinearModule(in_features=in_dim, out_features=out_dim, activation=act,
                                         norm=norm, dropout_p=drop_p, weight_init=weight_init, use_noisy=use_noisy)
            self.layers.append(linear_module)

        output_layer = LinearModule(in_features=input_dims[-1], out_features=output_dims[-1],
                                    activation=out_activation,
                                    norm=None, dropout_p=0.0, weight_init=weight_init, use_noisy=use_noisy)
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
