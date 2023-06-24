import torch
import torch.nn as nn


NONLINEARITIES = {
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid()
}


class ConcatSquashLinearTrainable(nn.Module):
    def __init__(self, dim_in, dim_out, extra_dim=True):
        super(ConcatSquashLinearTrainable, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)
        
        self.extra_dim = extra_dim

    def forward(self, t, x):
        t = t.view(len(t), 1, 1, 1)
        
        if self.extra_dim:
            return self._layer(x) * torch.sigmoid(self._hyper_gate(t)) \
                + torch.tanh(self._hyper_bias(t))
        else:
            return self._layer(x) * torch.sigmoid(self._hyper_gate(t)) \
                + torch.tanh(self._hyper_bias(t))
            


class MLPConcatSquashTrainable(nn.Module):

    def __init__(self, dims, nonlinearity="softplus", final_activation=None,
                 extra_dim=True):

        super(MLPConcatSquashTrainable, self).__init__()

        self.num_layers = len(dims)

        layers = []

        activations = []

        for i in range(self.num_layers):

            layers.append(ConcatSquashLinearTrainable(dims[i][0], dims[i][1],
                                                      extra_dim=extra_dim))

            if i < self.num_layers - 1:

                activations.append(NONLINEARITIES[nonlinearity])

        self.layers = nn.ModuleList(layers)

        self.activations = nn.ModuleList(activations)

        self.final_activation = final_activation

        if not final_activation is None:
            self.final_activation_layer = NONLINEARITIES[final_activation]

    def forward(self, t, x):

        for i in range(self.num_layers):

            x = self.layers[i](t, x)

            if i < self.num_layers - 1:

                x = self.activations[i](x)

        if self.final_activation != None:

            x = self.final_activation_layer(x)

        return x


class MLP(nn.Module):

    def __init__(self, dims, nonlinearity="relu", final_activation=None):

        super(MLP, self).__init__()

        num_layers = len(dims)

        layers = []

        activations = []

        for i in range(num_layers):

            layers.append(nn.Linear(dims[i][0], dims[i][1]))

            if i < num_layers - 1:

                activations.append(NONLINEARITIES[nonlinearity])

        self.layers = nn.ModuleList(layers)

        self.activations = nn.ModuleList(activations)

        self.final_activation = final_activation

        if not final_activation is None:
            self.final_activation_layer = NONLINEARITIES[final_activation]
        
        self.num_layers = num_layers

    def forward(self, x):

        for i in range(self.num_layers):

            x = self.layers[i](x)

            if i < self.num_layers - 1:

                x = self.activations[i](x)

        if self.final_activation != None:

            x = self.final_activation_layer(x)

        return x
