import collections
import torch
import pandas as pd
from torch import nn
from torch.nn.utils import prune

from binn.model.pathway_network import dataframes_to_pathway_network
from binn.model.util import load_reactome_db


class BINN(nn.Module):
    """
    A biologically informed neural network (BINN) in pure PyTorch.

    If `heads_ensemble=False`, we build a standard sequential network
    with layer-to-layer connections.

    If `heads_ensemble=True`, we build an 'ensemble of heads' network:
    each hidden layer also produces a separate head (dimension = n_outputs)
    which is passed through a sigmoid, then summed at the end.

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
        n_layers (int, optional):
            Number of layers in the network (depth). Defaults to 4.
        n_outputs (int, optional):
            Dimension of the final output (e.g., 2 for binary classification). Defaults to 2.
        dropout (float, optional):
            Dropout probability. Defaults to 0.
        heads_ensemble (bool, optional):
            If True, build an ensemble-of-heads network. Otherwise, a standard MLP.
        device (str, optional):
            The PyTorch device to place this model on. Defaults to "cpu".

    Attributes:
        inputs (List[str]):
            The list of input feature names derived from the first connectivity matrix.
        layers (nn.Module):
            The built network (either standard sequential or ensemble-of-heads).
        layer_names (List[List[str]]):
            The node (feature) names for each layer, for interpretability.
        connectivity_matrices (List[pd.DataFrame]):
            The adjacency (pruning) masks for each layer, derived from the pathway network.
    """

    def __init__(
        self,
        data_matrix: pd.DataFrame = None,
        network_source: str = None,
        input_source: str = "uniprot",
        mapping: pd.DataFrame = None,
        pathways: pd.DataFrame = None,
        activation: str = "tanh",
        n_layers: int = 4,
        n_outputs: int = 2,
        dropout: float = 0,
        heads_ensemble: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device
        self.to(self.device)

        self.n_layers = n_layers
        self.heads_ensemble = heads_ensemble

        # Build the pathway network from dataframes
  
        if network_source == "reactome":
            reactome_db = load_reactome_db(input_source=input_source)
            mapping = reactome_db["mapping"]
            pathways = reactome_db["pathways"]

        # Build connectivity from the pathway network
        pn = dataframes_to_pathway_network(
            data_matrix=data_matrix, pathway_df=pathways, mapping_df=mapping
        )

        # The connectivity matrices for each layer
        self.connectivity_matrices = pn.get_connectivity_matrices(n_layers=n_layers)

        # Collect layer sizes
        layer_sizes = []
        self.layer_names = []

        # First matrix => input layer size
        mat_first = self.connectivity_matrices[0]
        in_features, _ = mat_first.shape
        layer_sizes.append(in_features)

        self.inputs = mat_first.index.tolist()  # feature names
        self.layer_names.append(mat_first.index.tolist())

        # Additional layers
        for mat in self.connectivity_matrices[1:]:
            i, _ = mat.shape
            layer_sizes.append(i)
            self.layer_names.append(mat.index.tolist())

        # Build actual layers
        if heads_ensemble:
            self.layers = _generate_ensemble_of_heads(
                layer_sizes,
                self.connectivity_matrices,
                activation=activation,
                n_outputs=n_outputs,
                bias=True,
            )
        else:
            self.layers = _generate_sequential(
                layer_sizes,
                self.connectivity_matrices,
                activation=activation,
                n_outputs=n_outputs,
                dropout=dropout,
                bias=True,
            )

        # Weight init
        self.apply(_init_weights)

        # Print device info
        print(f"\n[INFO] BINN is on device: {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass; if heads_ensemble=True, sum-of-heads is used."""
        return self.layers(x)


def _init_weights(m):
    """Initialize Linear layers with Xavier uniform."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _append_activation(layers, activation, i):
    if activation == "tanh":
        layers.append((f"Tanh_{i}", nn.Tanh()))
    elif activation == "relu":
        layers.append((f"ReLU_{i}", nn.ReLU()))
    elif activation == "leaky relu":
        layers.append((f"LeakyReLU_{i}", nn.LeakyReLU()))
    elif activation == "sigmoid":
        layers.append((f"Sigmoid_{i}", nn.Sigmoid()))
    elif activation == "elu":
        layers.append((f"Elu_{i}", nn.ELU()))
    elif activation == "hardsigmoid":
        layers.append((f"HardSigmoid_{i}", nn.Hardsigmoid()))
    return layers


def _generate_sequential(
    layer_sizes,
    connectivity_matrices=None,
    activation: str = "tanh",
    n_outputs: int = 2,
    dropout: float = 0,
    bias: bool = True,
):
    """
    Standard MLP layers, each with optional pruning from connectivity_matrices,
    plus a final output layer of size n_outputs.
    """
    import collections

    layers = []
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        lin = nn.Linear(in_size, out_size, bias=bias)
        layers.append((f"Layer_{i}", lin))
        layers.append((f"BatchNorm_{i}", nn.BatchNorm1d(out_size)))

        # Prune if a connectivity matrix is provided
        if connectivity_matrices is not None:
            mask = torch.tensor(connectivity_matrices[i].T.values, dtype=torch.float32)
            prune.custom_from_mask(lin, name="weight", mask=mask)

        # Dropout
        layers.append((f"Dropout_{i}", nn.Dropout(dropout)))

        # Activation
        _append_activation(layers, activation, i)

    # Final output layer
    layers.append(("Output", nn.Linear(layer_sizes[-1], n_outputs, bias=bias)))

    return nn.Sequential(collections.OrderedDict(layers))


def _generate_ensemble_of_heads(
    layer_sizes,
    connectivity_matrices=None,
    activation="tanh",
    n_outputs=2,
    bias=True,
):
    """
    Build a multi-head ensemble: each hidden layer also has a head -> n_outputs,
    which is passed through Sigmoid, and then the heads are summed in forward pass.
    """
    return _EnsembleHeads(layer_sizes, connectivity_matrices, activation, n_outputs, bias)


class _EnsembleHeads(nn.Module):
    """An ensemble-of-heads approach with sums of sigmoids from each layer."""

    def __init__(self, layer_sizes, connectivity_matrices, activation, n_outputs, bias):
        super().__init__()
        import collections

        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.n_outputs = n_outputs

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            # Main block
            block_layers = []
            lin = nn.Linear(in_size, out_size, bias=bias)
            block_layers.append((f"Linear_{i}", lin))
            block_layers.append((f"BatchNorm_{i}", nn.BatchNorm1d(out_size)))

            # Prune
            if connectivity_matrices is not None:
                mask = torch.tensor(connectivity_matrices[i].T.values, dtype=torch.float32)
                prune.custom_from_mask(lin, name="weight", mask=mask)

            # Activation
            _append_activation(block_layers, activation, i)

            block = nn.Sequential(collections.OrderedDict(block_layers))
            self.blocks.append(block)

            # Head = (linear -> sigmoid)
            head_lin = nn.Linear(out_size, n_outputs, bias=bias)
            head = nn.Sequential(head_lin, nn.Sigmoid())
            self.heads.append(head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sum_heads = None

        for i, (block, head) in enumerate(zip(self.blocks, self.heads)):
            x = block(x)            # transform
            out = head(x)           # shape [batch, n_outputs]
            sum_heads = out if sum_heads is None else sum_heads + out

        return sum_heads
