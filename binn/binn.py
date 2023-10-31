import collections

import torch
import lightning as L
from torch import nn as nn
from torch.nn.utils import prune as prune

from binn.network import Network


class BINN(L.pytorch.LightningModule):
    """
    Implements a Biologically Informed Neural Network (BINN). The BINN
    is implemented using the Lightning-framework.
    If you are unfamiliar with PyTorch, we suggest visiting
    their website: https://pytorch.org/


    Args:
        pathways (Network): A Network object that defines the network topology.
        activation (str, optional): Activation function to use. Defaults to "tanh".
        weight (torch.Tensor, optional): Weights for loss function. Defaults to torch.Tensor([1, 1]).
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        n_layers (int, optional): Number of layers in the network. Defaults to 4.
        scheduler (str, optional): Learning rate scheduler to use. Defaults to "plateau".
        optimizer (str, optional): Optimizer to use. Defaults to "adam".
        validate (bool, optional): Whether to use validation data during training. Defaults to False.
        n_outputs (int, optional): Number of output nodes. Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.
        residual (bool, optional): Whether to use residual connections. Defaults to False.

    Attributes:
        residual (bool): Whether to use residual connections.
        pathways (Network): A Network object that defines the network topology.
        n_layers (int): Number of layers in the network.
        layer_names (List[str]): List of layer names.
        features (Index): A pandas Index object containing the input features.
        layers (nn.Module): The layers of the BINN.
        loss (nn.Module): The loss function used during training.
        learning_rate (float): Learning rate for optimizer.
        scheduler (str): Learning rate scheduler used.
        optimizer (str): Optimizer used.
        validate (bool): Whether to use validation data during training.
    """

    def __init__(
        self,
        network: Network = None,
        connectivity_matrices: list = None,
        activation: str = "tanh",
        weight: torch.tensor = torch.tensor([1, 1]),
        learning_rate: float = 1e-4,
        n_layers: int = 4,
        scheduler: str = "plateau",
        optimizer: str = "adam",
        validate: bool = False,
        n_outputs: int = 2,
        dropout: float = 0,
        residual: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.to(device)
        self.residual = residual
        if not connectivity_matrices:
            self.network = network
            self.connectivity_matrices = self.network.get_connectivity_matrices(
                n_layers
            )
        else:
            self.connectivity_matrices = connectivity_matrices
        self.n_layers = n_layers

        layer_sizes = []
        self.layer_names = []

        matrix = self.connectivity_matrices[0]
        i, _ = matrix.shape
        layer_sizes.append(i)
        self.layer_names.append(matrix.index.tolist())
        self.features = matrix.index
        self.trainable_params = matrix.to_numpy().sum()
        for matrix in self.connectivity_matrices[1:]:
            self.trainable_params += matrix.to_numpy().sum()
            i, _ = matrix.shape
            layer_sizes.append(i)
            self.layer_names.append(matrix.index.tolist())

        if self.residual:
            self.layers = _generate_residual(
                layer_sizes,
                connectivity_matrices=self.connectivity_matrices,
                activation="tanh",
                bias=True,
                n_outputs=2,
            )
        else:
            self.layers = _generate_sequential(
                layer_sizes,
                connectivity_matrices=self.connectivity_matrices,
                activation=activation,
                bias=True,
                n_outputs=n_outputs,
                dropout=dropout,
            )
        self.apply(_init_weights)
        self.weight = weight
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.validate = validate
        self.save_hyperparameters()
        print("\nBINN is on the device:", self.device, end="\n")

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Performs a forward pass through the BINN.

        Args:
            x (torch.Tensor): The input tensor to the BINN.

        Returns:
            torch.Tensor: The output tensor of the BINN.
        """
        if self.residual:
            return self._forward_residual(x)
        else:
            return self.layers(x)

    def training_step(self, batch, _):
        """
        Performs a single training step for the BINN.

        Args:
            batch: The batch of data to use for the training step.
            _: Not used.

        Returns:
            torch.Tensor: The loss tensor for the training step.
        """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x).to(self.device)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        """
        Implements a single validation step for the BINN.

        Args:
            batch: A tuple containing the input and output data for the current batch.
            _: The batch index, which is not used.
        """

        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": accuracy}

    def test_step(self, batch, _):
        """
        Implements a single testing step for the BINN.

        Args:
            batch: A tuple containing the input and output data for the current batch.
            _: The batch index, which is not used.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training the BINN.

        Returns:
            A list of optimizers and a list of learning rate schedulers.
        """
        if self.validate:
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
                    optimizer, patience=5, threshold=0.01, mode="min", verbose=True
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

    def calculate_accuracy(self, y, prediction) -> float:
        return torch.sum(y == prediction).item() / float(len(y))

    def get_connectivity_matrices(self) -> list:
        """
        Returns the connectivity matrices underlying the BINN.

        Returns:
            The connectivity matrices as a list of Pandas DataFrames.
        """
        return self.connectivity_matrices

    def reset_params(self):
        """
        Resets the trainable parameters of the BINN.
        """
        self.apply(_reset_params)

    def init_weights(self):
        """
        Initializes the trainable parameters of the BINN.
        """
        self.apply(_init_weights)


    def _forward_residual(self, x: torch.tensor):
        x_final = torch.tensor([0, 0], device=self.device)
        residual_counter: int = 0
        for name, layer in self.layers.named_children():
            if name.startswith("Residual"):
                if "out" in name:
                    x_temp = layer(x)
                if _is_activation(layer):
                    x_temp = layer(x_temp)
                    x_final = x_temp + x_final
                    residual_counter = residual_counter + 1
            else:
                x = layer(x)
        x_final = x_final / residual_counter
        return x_final


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
        layers.append((f"HardSigmoid {n}", nn.Hardsigmoid()))
    return layers


def _generate_sequential(
    layer_sizes,
    connectivity_matrices=None,
    activation: str = "tanh",
    bias: bool = True,
    n_outputs: int = 2,
    dropout: int = 0,
):
    layers = []
    for n in range(len(layer_sizes) - 1):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))  # linear layer
        layers.append((f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n + 1])))
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
    layers.append(("Output layer", nn.Linear(layer_sizes[-1], n_outputs, bias=bias)))
    model = nn.Sequential(collections.OrderedDict(layers))
    return model


def _generate_residual(
    layer_sizes, connectivity_matrices=None, activation="tanh", bias=False, n_outputs=2
):
    layers = []

    def generate_block(n, layers):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))
        layers.append((f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n + 1])))
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
            layers.append(("BatchNorm_final", nn.BatchNorm1d(layer_sizes[-1])))
            layers.append(("Dropout_final", nn.Dropout(0.2)))
            layers.append(
                (
                    "Residual_out_final",
                    nn.Linear(layer_sizes[-1], n_outputs, bias=bias),
                )
            )
            layers.append(("Residual_sigmoid_final", nn.Sigmoid()))
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
