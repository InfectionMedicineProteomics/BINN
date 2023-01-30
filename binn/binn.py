import collections

import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn.utils import prune as prune

from binn.network import Network


class BINN(LightningModule):
    def __init__(
        self,
        pathways: Network = None,
        activation: str = "tanh",
        weight: torch.Tensor = torch.Tensor([1, 1]),
        learning_rate: float = 1e-4,
        n_layers: int = 4,
        scheduler="plateau",
        optimizer="adam",
        validate: bool = False,
        n_outputs: int = 2,
        dropout: float = 0,
        residual: bool = False,
    ):

        super().__init__()
        self.residual = residual
        self.RN = pathways
        self.n_layers = n_layers

        connectivity_matrices = self.RN.get_connectivity_matrices(n_layers)
        layer_sizes = []
        self.layer_names = []

        for matrix in connectivity_matrices:
            i, _ = matrix.shape
            layer_sizes.append(i)
            self.layer_names.append(matrix.index)

        if self.residual:
            self.layers = generate_residual(
                layer_sizes,
                connectivity_matrices=connectivity_matrices,
                activation="tanh",
                bias=True,
                n_outputs=2,
            )
        else:
            self.layers = generate_sequential(
                layer_sizes,
                connectivity_matrices=connectivity_matrices,
                activation=activation,
                bias=True,
                n_outputs=n_outputs,
                dropout=dropout,
            )
        init_weights(self.layers)
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.validate = validate
        self.save_hyperparameters()

    def forward(self, x):
        if self.residual:
            return forward_residual(self.layers, x)
        else:
            return self.layers(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def report_layer_structure(self, verbose=False):
        if verbose:
            print(self.layers)
        parameters = {"nz weights": [], "weights": [], "biases": []}
        for i, l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                nz_weights = torch.count_nonzero(l.weight)
                weights = torch.numel(l.weight)
                biases = torch.numel(l.bias)
                if verbose:
                    print(f"Layer {i}")
                    print(f"Number of nonzero weights: {nz_weights} ")
                    print(f"Number biases: {nz_weights} ")
                    print(f"Total number of elements: {weights + biases} ")
                parameters["nz weights"].append(nz_weights)
                parameters["weights"].append(weights)
                parameters["biases"].append(biases)
        return parameters

    def configure_optimizers(self):
        if self.validate == True:
            monitor = "val_loss"
        else:
            monitor = "train_loss"

        if isinstance(self.optimizer, str):
            if self.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=self.learning_rate, weight_decay=1e-3
                )
        else:
            optimizer = self.optimizer

        if self.scheduler == "plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, threshold=0.00001, mode="min", verbose=True
                ),
                "interval": "epoch",
                "monitor": monitor,
            }
        elif self.scheduler == "step":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=25, gamma=0.1, verbose=True
                )
            }

        return [optimizer], [scheduler]

    def calculate_accuracy(self, y, prediction):
        return torch.sum(y == prediction).item() / (float(len(y)))

    def get_connectivity_matrices(self):
        return self.RN.get_connectivity_matrices(self.n_layers)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


def reset_params(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def append_activation(layers, activation, n):
    if activation == "tanh":
        layers.append((f"Tanh {n}", nn.Tanh()))
    elif activation == "relu":
        layers.append((f"ReLU {n}", nn.ReLU()))
    elif activation == "leaky relu":
        layers.append((f"LeakyReLU {n}", nn.LeakyReLU()))
    elif activation == "sigmoid":
        layers.append((f"Sigmoid {n}", nn.Sigmoid()))
    elif activation == "elu":
        layers.append((f"Elu {n}", nn.ELU()))
    elif activation == "hardsigmoid":
        layers.append((f"HardSigmoid", nn.Hardsigmoid()))
    return layers


def generate_sequential(
    layer_sizes,
    connectivity_matrices=None,
    activation="tanh",
    bias=True,
    n_outputs=2,
    dropout=0,
):
    """
    Generates a sequential model from layer sizes.
    """
    layers = []
    for n in range(len(layer_sizes) - 1):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))  # linear layer
        layers.append(
            (f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n + 1]))
        )  # batch normalization
        if connectivity_matrices is not None:
            # Masking matrix
            prune.custom_from_mask(
                linear_layer,
                name="weight",
                mask=torch.tensor(connectivity_matrices[n].T.values),
            )
        if isinstance(dropout, list):
            layers.append((f"Dropout_{n}", nn.Dropout(dropout[n])))
        else:
            layers.append((f"Dropout_{n}", nn.Dropout(dropout)))
        if isinstance(activation, list):
            layers.append((f"Activation_{n}", activation[n]))
        else:
            append_activation(layers, activation, n)
    layers.append(
        ("Output layer", nn.Linear(layer_sizes[-1], n_outputs, bias=bias))
    )  # Output layer
    model = nn.Sequential(collections.OrderedDict(layers))
    return model


def generate_residual(
    layer_sizes, connectivity_matrices=None, activation="tanh", bias=False, n_outputs=2
):
    layers = []

    def generate_block(n, layers):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))  # linear layer
        layers.append(
            (f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n + 1]))
        )  # batch normalization
        if connectivity_matrices is not None:
            prune.custom_from_mask(
                linear_layer,
                name="weight",
                mask=torch.tensor(connectivity_matrices[n].T.values),
            )
        layers.append((f"Dropout_{n}", nn.Dropout(0.2)))
        layers.append(
            (f"Residual_out_{n}", nn.Linear(layer_sizes[n + 1], n_outputs, bias=bias))
        )
        layers.append((f"Residual_sigmoid_{n}", nn.Sigmoid()))
        return layers

    for res_index in range(len(layer_sizes)):
        if res_index == len(layer_sizes) - 1:
            layers.append(
                (f"BatchNorm_final", nn.BatchNorm1d(layer_sizes[-1]))
            )  # batch normalization
            layers.append((f"Dropout_final", nn.Dropout(0.2)))
            layers.append(
                (
                    f"Residual_out_final",
                    nn.Linear(layer_sizes[-1], n_outputs, bias=bias),
                )
            )
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
    elif isinstance(layer, nn.Sigmoid):
        return True
    return False


def forward_residual(model: nn.Sequential, x):
    x_final = torch.Tensor([0, 0])
    residual_counter = 0
    for name, layer in model.named_children():
        if name.startswith("Residual"):
            if "out" in name:  # we've reached res output
                x_temp = layer(x)
            if is_activation(
                layer
            ):  # weve reached output sigmoid. Time to add to x_final
                x_temp = layer(x_temp)
                x_final = x_final + x_temp
                residual_counter = residual_counter + 1
        else:
            x = layer(x)
    x_final = x_final / (residual_counter)  # average

    return x_final
