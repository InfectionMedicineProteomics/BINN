
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
