import collections

import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn.utils import prune as prune

from binn.network import Network


class BINN(LightningModule):
    """
    The core BINN module. This inherits from the PyTorch Lightning LightningModule.
    """

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
        self.pathways = pathways
        self.n_layers = n_layers

        connectivity_matrices = self.pathways.get_connectivity_matrices(
            n_layers)
        layer_sizes = []
        self.layer_names = []

        matrix = connectivity_matrices[0]
        i, _ = matrix.shape
        layer_sizes.append(i)
        self.layer_names.append(matrix.index)
        self.features = matrix.index
        for matrix in connectivity_matrices[1:]:
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
            self.layers = _generate_sequential(
                layer_sizes,
                connectivity_matrices=connectivity_matrices,
                activation=activation,
                bias=True,
                n_outputs=n_outputs,
                dropout=dropout,
            )
        self.apply(_init_weights)
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.validate = validate
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.
        """
        if self.residual:
            return _forward_residual(self.layers, x)
        else:
            return self.layers(x)

    def training_step(self, batch, _):
        """
        Training step.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("train_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        """
        Validation step.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True,
                 on_step=False, on_epoch=True)

    def test_step(self, batch, _):
        """
        Test step.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("test_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        self.log("test_acc", accuracy, prog_bar=True,
                 on_step=False, on_epoch=True)

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
        """
        Returns the connectivity matrices underlying the BINN.
        """
        return self.pathways.get_connectivity_matrices(self.n_layers)

    def reset_params(self):
        """
        Resets trainable parameters.
        """
        self.apply(_reset_params)

    def init_weights(self):
        self.apply(_init_weights)


def _init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def _reset_params(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def _append_activation(layers, activation, n):
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


def _generate_sequential(
    layer_sizes,
    connectivity_matrices=None,
    activation="tanh",
    bias=True,
    n_outputs=2,
    dropout=0,
):

    layers = []
    for n in range(len(layer_sizes) - 1):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))  # linear layer
        layers.append(
            (f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n + 1]))
        )
        if connectivity_matrices is not None:
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
            _append_activation(layers, activation, n)
    layers.append(
        ("Output layer", nn.Linear(layer_sizes[-1], n_outputs, bias=bias))
    )
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
            (f"Residual_out_{n}", nn.Linear(
                layer_sizes[n + 1], n_outputs, bias=bias))
        )
        layers.append((f"Residual_sigmoid_{n}", nn.Sigmoid()))
        return layers

    for res_index in range(len(layer_sizes)):
        if res_index == len(layer_sizes) - 1:
            layers.append(
                (f"BatchNorm_final", nn.BatchNorm1d(layer_sizes[-1]))
            )
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
            _append_activation(layers, activation, res_index)

    model = nn.Sequential(collections.OrderedDict(layers))
    return model


def _is_activation(layer):
    if isinstance(layer, nn.Tanh):
        return True
    elif isinstance(layer, nn.ReLU):
        return True
    elif isinstance(layer, nn.LeakyReLU):
        return True
    elif isinstance(layer, nn.Sigmoid):
        return True
    return False


def _forward_residual(model: nn.Sequential, x):
    x_final = torch.Tensor([0, 0])
    residual_counter = 0
    for name, layer in model.named_children():
        if name.startswith("Residual"):
            if "out" in name:
                x_temp = layer(x)
            if _is_activation(
                layer
            ):
                x_temp = layer(x_temp)
                x_final = x_final + x_temp
                residual_counter = residual_counter + 1
        else:
            x = layer(x)
    x_final = x_final / (residual_counter)

    return x_final
