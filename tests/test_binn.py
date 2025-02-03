import pytest
import torch
import pandas as pd
import numpy as np


from binn.model.binn import BINN


def get_dummy_connectivity_matrices(n_layers: int):
    """
    Create a list of dummy connectivity matrices as pandas DataFrames.

    For example, let’s assume n_layers=3. Then we create:
      - Matrix 0: shape (5, 3) → used for the input layer.
      - Matrix 1: shape (3, 2).
      - Matrix 2: shape (2, 2).
    """
    # Matrix 0: input layer: 5 features → 3 hidden nodes
    mat0 = pd.DataFrame(
        np.ones((5, 3)),
        index=[f"in{i}" for i in range(1, 6)],
        columns=[f"h1_{j}" for j in range(1, 4)],
    )
    # Matrix 1: 3 nodes → 2 nodes
    mat1 = pd.DataFrame(
        np.ones((3, 2)),
        index=[f"h1_{j}" for j in range(1, 4)],
        columns=[f"h2_{k}" for k in range(1, 3)],
    )
    # Matrix 2: 2 nodes → 2 nodes
    mat2 = pd.DataFrame(
        np.ones((2, 2)),
        index=[f"h2_{k}" for k in range(1, 3)],
        columns=[f"h3_{m}" for m in range(1, 3)],
    )
    return [mat0, mat1, mat2]


class DummyPathwayNetwork:
    """A dummy class mimicking the object returned by dataframes_to_pathway_network()."""

    def __init__(self, connectivity_matrices):
        self._connectivity_matrices = connectivity_matrices

    def get_connectivity_matrices(self, n_layers: int):
        # For testing we simply return our dummy matrices regardless of n_layers.
        return self._connectivity_matrices


# === Pytest Fixtures to Monkey-Patch External Dependencies ===


@pytest.fixture(autouse=True)
def dummy_dataframes_to_pathway_network(monkeypatch):
    """
    Override the dataframes_to_pathway_network function so that BINN always uses the dummy pathway network.
    """
    from binn.model import binn  # import the module where BINN lives

    def dummy_dataframes_to_pathway_network_func(*args, **kwargs):
        mats = get_dummy_connectivity_matrices(n_layers=3)
        return DummyPathwayNetwork(mats)

    monkeypatch.setattr(
        binn, "dataframes_to_pathway_network", dummy_dataframes_to_pathway_network_func
    )


@pytest.fixture
def dummy_load_reactome_db(monkeypatch):
    """
    Override load_reactome_db so that when network_source="reactome" is used,
    a known dummy mapping & pathways DataFrame are returned.
    """
    from binn.model import binn  # import the module where BINN lives

    # Create dummy mapping and pathways DataFrames.
    mapping_df = pd.DataFrame(
        {
            "input": ["A", "B", "C"],
            "translation": ["T_A", "T_B", "T_C"],
            "url": ["url1", "url2", "url3"],
            "name": ["Pathway A", "Pathway B", "Pathway C"],
            "x": [0, 0, 0],
            "species": ["human", "human", "human"],
        }
    )
    pathways_df = pd.DataFrame({"source": ["A", "B"], "target": ["B", "C"]})
    dummy_db = {"mapping": mapping_df, "pathways": pathways_df}
    monkeypatch.setattr(binn, "load_reactome_db", lambda input_source: dummy_db)
    return dummy_db


# === Tests ===


def test_sequential_forward():
    """
    Test the standard (sequential) BINN:
      - Build a dummy data_matrix (10 samples, 5 features).
      - Build a BINN with n_layers=3 and heads_ensemble=False.
      - Run a forward pass and check that the output shape is (batch_size, n_outputs).
    """
    # Create dummy data_matrix (the connectivity matrices will determine the input size)
    data = np.random.rand(10, 5)
    data_matrix = pd.DataFrame(data, columns=[f"in{i}" for i in range(1, 6)])

    model = BINN(
        data_matrix=data_matrix,
        network_source=None,
        mapping=pd.DataFrame(),  # provided but not used because we monkey-patched connectivity
        pathways=pd.DataFrame(),
        n_layers=3,
        n_outputs=2,
        heads_ensemble=False,
        device="cpu",
    )

    # Create a dummy input tensor with the same feature dimension (5)
    x = torch.randn(4, 5)
    output = model(x)

    # Expect the output shape to be (4, 2)
    assert output.shape == (4, 2)


def test_heads_ensemble_forward():
    """
    Test the ensemble-of-heads BINN:
      - Build a dummy data_matrix.
      - Build a BINN with heads_ensemble=True.
      - Run a forward pass and verify the output shape.
    """
    data = np.random.rand(10, 5)
    data_matrix = pd.DataFrame(data, columns=[f"in{i}" for i in range(1, 6)])

    model = BINN(
        data_matrix=data_matrix,
        network_source=None,
        mapping=pd.DataFrame(),
        pathways=pd.DataFrame(),
        n_layers=3,
        n_outputs=2,
        heads_ensemble=True,
        device="cpu",
    )

    x = torch.randn(4, 5)
    output = model(x)
    # The ensemble method should still produce an output of shape (batch_size, n_outputs)
    assert output.shape == (4, 2)


def test_device_assignment():
    """
    Test that after initialization the model’s parameters are on the proper device.
    """
    data = np.random.rand(5, 5)
    data_matrix = pd.DataFrame(data, columns=[f"in{i}" for i in range(1, 6)])
    device = "cpu"

    model = BINN(
        data_matrix=data_matrix,
        network_source=None,
        mapping=pd.DataFrame(),
        pathways=pd.DataFrame(),
        n_layers=3,
        n_outputs=2,
        heads_ensemble=False,
        device=device,
    )

    # Check that all parameters are on the requested device.
    for param in model.parameters():
        assert param.device.type == device


def test_initialization_with_reactome_source(dummy_load_reactome_db):
    """
    Test that if network_source is set to "reactome", then the model correctly loads the dummy reactome DB.
    """
    data = np.random.rand(5, 5)
    data_matrix = pd.DataFrame(data, columns=[f"in{i}" for i in range(1, 6)])

    model = BINN(
        data_matrix=data_matrix,
        network_source="reactome",
        input_source="dummy",  # not used in dummy_load_reactome_db, but required by the signature
        mapping=None,
        pathways=None,
        n_layers=3,
        n_outputs=2,
        heads_ensemble=False,
        device="cpu",
    )

    # Check that connectivity_matrices are set and have the expected number of layers.
    assert isinstance(model.connectivity_matrices, list)
    assert len(model.connectivity_matrices) == 3
