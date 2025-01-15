import collections

import torch
import lightning.pytorch as pl
import pandas as pd
from torch import nn
from torch.nn.utils import prune

from binn.model.pathway_network import dataframes_to_pathway_network
from binn.model.util import load_reactome_db


class BINN(pl.LightningModule):
    """
    A biologically informed neural network (BINN) using PyTorch Lightning.
    - If `heads_ensemble=False`, we build a standard sequential network with layer-to-layer connections.
    - If `heads_ensemble=True`, we actually build an 'ensemble of heads' network:
      each layer produces a separate output head (same dimension = n_outputs),
      each head goes through a sigmoid, and at the end we sum them all.

    Args:
        data_matrix (pd.DataFrame, optional):
            A DataFrame of input features (samples x features). If not needed, can be None.
        use_reactome (bool, optional):
            If True, loads `mapping` and `pathways` from `load_reactome_db()`, ignoring the ones provided.
        mapping (pd.DataFrame, optional):
            A DataFrame describing how each input feature maps into the pathway graph.
            If None, the user must rely on `use_reactome=True`.
        pathways (pd.DataFrame, optional):
            A DataFrame describing the edges among pathway nodes.
        activation (str, optional):
            The activation function to use in each layer. Defaults to "tanh".
        weight (torch.Tensor, optional):
            Class weight for the loss function. Defaults to `tensor([1, 1])`.
        learning_rate (float, optional):
            Learning rate for the optimizer. Defaults to 1e-4.
        n_layers (int, optional):
            Number of layers in the network (i.e., the depth of the BINN). Defaults to 4.
        scheduler (str, optional):
            Learning rate scheduler, e.g. "plateau" or "step". Defaults to "plateau".
        optimizer (str, optional):
            Optimizer name, e.g. "adam". Defaults to "adam".
        validate (bool, optional):
            Whether to monitor validation metrics. Defaults to False.
        n_outputs (int, optional):
            Dimension of the final output (e.g., 2 for binary classification). Defaults to 2.
        dropout (float, optional):
            Dropout probability. Defaults to 0.
        heads_ensemble (bool, optional):
            If True, build an ensemble-of-heads network. Otherwise, build a standard sequential network.
        device (str, optional):
            The PyTorch device to place this model on. Defaults to "cpu".

    Attributes:
        connectivity_matrices (List[pd.DataFrame]):
            The adjacency masks used for pruning each layer.
        layers (nn.Module):
            The built network (either standard sequential or ensemble-of-heads).
        layer_names (List[List[str]]):
            Names of the nodes in each layer (for interpretability).
        features (pd.Index):
            The set of input features (row index of the first connectivity matrix).
    """

    def __init__(
        self,
        data_matrix: pd.DataFrame = None,
        use_reactome: bool = False,
        mapping: pd.DataFrame = None,
        pathways: pd.DataFrame = None,
        activation: str = "tanh",
        weight: torch.Tensor = torch.tensor([1, 1]),
        learning_rate: float = 1e-4,
        n_layers: int = 4,
        scheduler: str = "plateau",
        optimizer: str = "adam",
        validate: bool = False,
        n_outputs: int = 2,
        dropout: float = 0,
        heads_ensemble: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.to(device)

        self.n_layers = n_layers
        self.heads_ensemble = heads_ensemble
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.validate = validate
        self.weight = weight
        self.loss = nn.CrossEntropyLoss(weight=weight)

        # Build the pathway network from dataframes
        if use_reactome:
            reactome_db = load_reactome_db()
            mapping = reactome_db["mapping"]
            pathways = reactome_db["pathways"]

        # Build the pathway-based connectivity
        pn = dataframes_to_pathway_network(
            data_matrix=data_matrix, pathway_df=pathways, mapping_df=mapping
        )
        # Connectivity matrices for each layer
        self.connectivity_matrices = pn.get_connectivity_matrices(n_layers=n_layers)

        # Keep track of layer sizes & names
        layer_sizes = []
        self.layer_names = []

        # The first matrix defines the input layer
        mat_first = self.connectivity_matrices[0]
        self.inputs = mat_first.index.tolist()
        in_features, _ = mat_first.shape
        layer_sizes.append(in_features)
        self.layer_names.append(mat_first.index.tolist())
        self.features = mat_first.index

        # For debugging: track a "rough" param count
        self.trainable_params = mat_first.to_numpy().sum() + len(mat_first.index)

        # Additional layers
        for mat in self.connectivity_matrices[1:]:
            self.trainable_params += mat.to_numpy().sum() + len(mat.index)
            i, _ = mat.shape
            layer_sizes.append(i)
            self.layer_names.append(mat.index.tolist())

        # Build either standard sequential or ensemble-of-heads
        if self.heads_ensemble:
            self.layers = _generate_ensemble_of_heads(
                layer_sizes,
                connectivity_matrices=self.connectivity_matrices,
                activation=activation,
                bias=True,
                n_outputs=n_outputs,
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

        # Initialize weights
        self.apply(_init_weights)

        self.save_hyperparameters()
        print("\nBINN is on the device:", self.device, end="\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BINN.

        If `self.heads_ensemble==True`, it calls the ensemble-of-heads forward
        (i.e. sums all heads). Otherwise, it just calls `self.layers(x)`.
        """
        return self.layers(x)

    def training_step(self, batch, _):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.calculate_accuracy(y, pred)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.calculate_accuracy(y, pred)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.calculate_accuracy(y, pred)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """
        Defines the optimizer & LR scheduler.
        """
        if self.validate:
            monitor = "val_loss"
        else:
            monitor = "train_loss"

        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=self.learning_rate, weight_decay=1e-3
                )
                self.optimizer = optimizer
        else:
            optimizer = self.optimizer

        if self.scheduler == "plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, threshold=0.01, mode="min"
                ),
                "interval": "epoch",
                "monitor": monitor,
            }
        elif self.scheduler == "step":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=25, gamma=0.1
                )
            }
        else:
            scheduler = None

        return ([optimizer], [scheduler]) if scheduler else ([optimizer], [])

    def calculate_accuracy(self, y, prediction) -> float:
        return torch.sum(y == prediction).item() / float(len(y))

    def get_connectivity_matrices(self) -> list:
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


#
# Helper functions
#


def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def _reset_params(m):
    if isinstance(m, (nn.BatchNorm1d, nn.Linear)):
        m.reset_parameters()


def _append_activation(layers, activation, layer_idx):
    if activation == "tanh":
        layers.append((f"Tanh {layer_idx}", nn.Tanh()))
    elif activation == "relu":
        layers.append((f"ReLU {layer_idx}", nn.ReLU()))
    elif activation == "leaky relu":
        layers.append((f"LeakyReLU {layer_idx}", nn.LeakyReLU()))
    elif activation == "sigmoid":
        layers.append((f"Sigmoid {layer_idx}", nn.Sigmoid()))
    elif activation == "elu":
        layers.append((f"Elu {layer_idx}", nn.ELU()))
    elif activation == "hardsigmoid":
        layers.append((f"HardSigmoid {layer_idx}", nn.Hardsigmoid()))
    return layers


def _generate_sequential(
    layer_sizes,
    connectivity_matrices=None,
    activation: str = "tanh",
    bias: bool = True,
    n_outputs: int = 2,
    dropout: float = 0,
):
    """
    Standard feed-forward multi-layer network with pruning from connectivity_matrices.
    Only the LAST layer outputs dimension = n_outputs.
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        lin = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias)
        layers.append((f"Layer_{i}", lin))
        layers.append((f"BatchNorm_{i}", nn.BatchNorm1d(layer_sizes[i + 1])))

        # Prune
        if connectivity_matrices is not None:
            mask = torch.tensor(connectivity_matrices[i].T.values, dtype=torch.float32)
            prune.custom_from_mask(lin, name="weight", mask=mask)

        # Dropout
        layers.append((f"Dropout_{i}", nn.Dropout(dropout)))

        # Activation
        _append_activation(layers, activation, i)

    # The output layer
    layers.append(("Output", nn.Linear(layer_sizes[-1], n_outputs, bias=bias)))
    model = nn.Sequential(collections.OrderedDict(layers))
    return model


def _generate_ensemble_of_heads(
    layer_sizes,
    connectivity_matrices=None,
    activation="tanh",
    bias=True,
    n_outputs=2,
):

    return EnsembleHeads(
        layer_sizes,
        connectivity_matrices=connectivity_matrices,
        activation=activation,
        bias=bias,
        n_outputs=n_outputs,
    )


class EnsembleHeads(nn.Module):
    """
    A network that processes input x layer by layer,
    collecting an "output head" at each layer.
    The final output is the sum of all heads (each passed through a Sigmoid).
    """

    def __init__(
        self,
        layer_sizes,
        connectivity_matrices=None,
        activation="tanh",
        bias=True,
        n_outputs=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.activation_name = activation

        for i in range(len(layer_sizes) - 1):
            # build the main transform block for layer i
            block = nn.Sequential()
            lin = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias)
            block.add_module(f"Linear_{i}", lin)
            block.add_module(f"BatchNorm_{i}", nn.BatchNorm1d(layer_sizes[i + 1]))

            if connectivity_matrices is not None:
                mask = torch.tensor(
                    connectivity_matrices[i].T.values, dtype=torch.float32
                )
                prune.custom_from_mask(lin, name="weight", mask=mask)

            # Activation
            block.add_module(f"Activation_{i}", _make_activation(activation))

            self.blocks.append(block)

            # Each layer i also gets a "head" from the layer output -> n_outputs
            head_lin = nn.Linear(layer_sizes[i + 1], n_outputs, bias=bias)
            head = nn.Sequential(head_lin, nn.Sigmoid())
            self.heads.append(head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We'll accumulate each head's output
        sum_of_heads = None

        for i, (block, head) in enumerate(zip(self.blocks, self.heads)):
            x = block(x)  # transform
            head_out = head(x)  # shape [batch_size, n_outputs]
            if sum_of_heads is None:
                sum_of_heads = head_out
            else:
                sum_of_heads = sum_of_heads + head_out

        return sum_of_heads


def _make_activation(activation_name: str) -> nn.Module:
    if activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "leaky relu":
        return nn.LeakyReLU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "elu":
        return nn.ELU()
    elif activation_name == "hardsigmoid":
        return nn.Hardsigmoid()
    else:
        return nn.Identity()
