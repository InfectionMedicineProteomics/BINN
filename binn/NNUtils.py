
import collections 
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch



def append_activation(layers, activation, n):
    if activation == 'tanh':
        layers.append((f'Tanh {n}', nn.Tanh()))
    elif activation == 'relu':
        layers.append((f'ReLU {n}', nn.ReLU()))
    elif activation == "leaky relu":
        layers.append((f'LeakyReLU {n}', nn.LeakyReLU()))
    elif activation == "sigmoid":
        layers.append((f'Sigmoid {n}', nn.Sigmoid()))
    elif activation == 'elu':
        layers.append((f'Elu {n}', nn.ELU()))
    elif activation == 'hardsigmoid':
        layers.append((f'HardSigmoid', nn.Hardsigmoid()))
    return layers
    

def generate_sequential(layer_sizes, 
                        connectivity_matrices = None, 
                        activation='tanh', bias=True,
                        n_outputs=2, dropout=0):
    """
    Generates a sequential model from layer sizes.
    """
    layers = []
    for n in range(len(layer_sizes)-1):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n+1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer)) # linear layer 
        layers.append((f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n+1]))) # batch normalization
        if connectivity_matrices is not None:
            # Masking matrix
            prune.custom_from_mask(linear_layer, name='weight', mask=torch.tensor(connectivity_matrices[n].T.values))
        if isinstance(dropout, list):
            layers.append((f"Dropout_{n}", nn.Dropout(dropout[n])))
        else:
            layers.append((f"Dropout_{n}", nn.Dropout(dropout)))
        if isinstance(activation, list):
            layers.append((f'Activation_{n}', activation[n]))
        else:
            append_activation(layers, activation, n)
    layers.append(("Output layer", nn.Linear(layer_sizes[-1],n_outputs, bias=bias))) # Output layer
    model = nn.Sequential(collections.OrderedDict(layers))
    return model


def generate_residual(layer_sizes,
                      connectivity_matrices=None,
                      activation='tanh', bias=False,
                      n_outpus = 2):
    layers = []

    def generate_block(n, layers):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n+1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))  # linear layer
        layers.append((f"BatchNorm_{n}", nn.BatchNorm1d(
            layer_sizes[n+1])))  # batch normalization
        if connectivity_matrices is not None:
            prune.custom_from_mask(linear_layer, name='weight', mask=torch.tensor(
                connectivity_matrices[n].T.values))
        layers.append((f"Dropout_{n}", nn.Dropout(0.2)))
        layers.append((f'Residual_out_{n}', nn.Linear(
            layer_sizes[n+1], n_outpus, bias=bias)))
        layers.append((f"Residual_sigmoid_{n}", nn.Sigmoid()))
        return layers

    for res_index in range(len(layer_sizes)):
        if res_index == len(layer_sizes)-1:
            layers.append((f"BatchNorm_final", nn.BatchNorm1d(
                layer_sizes[-1])))  # batch normalization
            layers.append((f"Dropout_final", nn.Dropout(0.2)))
            layers.append((f'Residual_out_final', nn.Linear(
                layer_sizes[-1], n_outpus, bias=bias)))
            layers.append((f"Residual_sigmoid_final", nn.Sigmoid()))
        else:
            layers = generate_block(res_index, layers)
            append_activation(layers, activation, res_index)

    model = nn.Sequential(collections.OrderedDict(layers))
    return model


def is_activation(layer):
    if isinstance(layer, nn.Tanh):
        return True
    elif isinstance(layer, nn.ReLU):
        return True
    elif isinstance(layer, nn.LeakyReLU):
        return True
    return False


def forward_residual(model: nn.Sequential, x):
    x_final = torch.Tensor([0, 0])
    residual_counter = 0
    for name, layer in model.named_children():
        if name.startswith("Residual"):
            if "out" in name:  # we've reached res output
                x_temp = layer(x)
            if "sigmoid" in name:  # weve reached output sigmoid. Time to add to x_final
                x_temp = layer(x_temp)
                x_final = x_final + x_temp
                residual_counter = residual_counter + 1
        else:
            x = layer(x)
    x_final = x_final / (residual_counter)  # average

    return x_final