import pytest
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from binn.analysis.explainer import BINNExplainer
from binn import BINNTrainer

class DummyModel(nn.Module):
    """
    A dummy model to mimic a trained BINN.
    It needs the following attributes:
      - device (a string, e.g., "cpu")
      - connectivity_matrices: a list of pandas DataFrames (one per layer)
      - layer_names: a list (of lists) of feature names (one per layer)
      - layers: a module whose named_children() returns at least one linear layer
    """
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        # Create one dummy connectivity matrix (for one hidden layer).
        # Assume the input layer has 2 nodes, and the next layer has 2 nodes.
        dummy_cm = pd.DataFrame(
            np.ones((2, 2)),
            index=["node1", "node2"],
            columns=["node1_out", "node2_out"],
        )
        self.connectivity_matrices = [dummy_cm]
        self.layer_names = [["node1", "node2"]]

        # Create a dummy layers module containing one Linear layer.
        # (BINNExplainer._explain_layers considers layers that are nn.Linear.)
        self.layers = nn.Sequential(nn.Linear(2, 2))

    def forward(self, x):
        return self.layers(x)


class DummyTrainer(BINNTrainer):
    """
    A dummy trainer that simulates training the model.
    The fit() method simply records the number of epochs and does nothing else.
    """
    def __init__(self):
        self.metrics = {}

    def update_model(self, model):
        self.model = model

    def fit(self, dataloaders, num_epochs):
        self.metrics["last_epochs"] = num_epochs
        # In a real trainer, training logic would be here.
        return


# --- Dummy DataLoaders --- #
# We simulate a DataLoader as an iterable of batches, where each batch is a tuple (inputs, targets).
def dummy_dataloader():
    # Create 3 batches; each batch has 2 samples.
    for _ in range(3):
        inputs = torch.randn(2, 2)  # 2 features matching DummyModel's input dimension.
        targets = torch.randint(0, 2, (2,))
        yield (inputs, targets)

dummy_dataloaders = {"train": list(dummy_dataloader()), "val": list(dummy_dataloader())}


# --- Monkey-Patching / Dummy SHAP Helper --- #
def dummy_explain_layers(self, background_data, test_data):
    """
    A dummy replacement for BINNExplainer._explain_layers.
    Returns a dictionary with:
       - 'features': a list (one per layer) of feature names,
       - 'shap_values': a list (one per layer) of dummy SHAP arrays.
    Creates a dummy SHAP array of shape (10, 2, 2) so that svals.mean(axis=0) yields a (2,2) array.
    """
    dummy_features = self.model.layer_names[0]
    dummy_shap = np.abs(np.random.rand(10, 2, 2))
    return {"features": [dummy_features], "shap_values": [dummy_shap]}


# --- Pytest Fixtures --- #
@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def explainer(dummy_model, monkeypatch):
    # Create a BINNExplainer instance using the dummy model.
    expl = BINNExplainer(dummy_model)
    # Monkey-patch its _explain_layers method with our dummy_explain_layers.
    monkeypatch.setattr(BINNExplainer, "_explain_layers", dummy_explain_layers)
    return expl


# --- Test Cases --- #
def test_gather_all_from_dataloader(dummy_model):
    """
    Test the _gather_all_from_dataloader helper:
      - With a specified split.
      - With all splits (when split is None).
    """
    expl = BINNExplainer(dummy_model)
    # Test with split "train"
    X_train, Y_train = expl._gather_all_from_dataloader(dummy_dataloaders, split="train")
    # With 3 batches of 2 samples each, expect 6 samples.
    assert X_train.shape[0] == 6
    assert Y_train.shape[0] == 6

    # Test with split=None (concatenating both "train" and "val")
    X_all, Y_all = expl._gather_all_from_dataloader(dummy_dataloaders, split=None)
    # Expect 12 samples total (6 from train + 6 from val)
    assert X_all.shape[0] == 12
    assert Y_all.shape[0] == 12


def test_normalize_importances(dummy_model):
    """
    Test the normalization logic in normalize_importances.
    Creates a small dummy explanation DataFrame and verifies that a normalized_importance column is added.
    Tests both "fan" and "subgraph" normalization methods.
    """
    expl = BINNExplainer(dummy_model)
    # Create a dummy explanation DataFrame with two edges.
    df = pd.DataFrame({
        "source_node": ["A", "B"],
        "target_node": ["B", "C"],
        "importance": [0.5, 1.0]
    })

    # Test using the "fan" method.
    df_fan = expl.normalize_importances(df, method="fan")
    assert "normalized_importance" in df_fan.columns

    # Test using the "subgraph" method.
    df_sub = expl.normalize_importances(df, method="subgraph")
    assert "normalized_importance" in df_sub.columns

    # Ensure that normalized_importance column values are numeric.
    assert pd.api.types.is_numeric_dtype(df_fan["normalized_importance"])
    assert pd.api.types.is_numeric_dtype(df_sub["normalized_importance"])


def test_explain_single_returns_dataframe(explainer):
    """
    Test explain_single:
      - Run the explainer using a given split ("train")
      - Verify that the returned DataFrame has the expected columns.
    """
    df_explain = explainer.explain_single(dummy_dataloaders, split="train", normalization_method=None)
    expected_cols = [
        "source_layer", "target_layer", "source_node", "target_node", "class_idx", "importance"
    ]
    for col in expected_cols:
        assert col in df_explain.columns


def test_explain_iterations(monkeypatch, dummy_model):
    """
    Test the explain(...) method over multiple iterations using a dummy trainer.
    Simulate multiple re-initializations/training runs and verify that the combined DataFrame
    includes per-iteration and aggregated importance columns.
    """
    expl = BINNExplainer(dummy_model)
    # Monkey-patch _explain_layers to always return our dummy output.
    monkeypatch.setattr(BINNExplainer, "_explain_layers", dummy_explain_layers)
    dummy_trainer = DummyTrainer()

    # Run explain for 2 iterations with 1 epoch each.
    combined_df = expl.explain(
        dataloaders=dummy_dataloaders,
        nr_iterations=2,
        num_epochs=1,
        trainer=dummy_trainer,
        split="train",
        normalization_method=None  # skip normalization in this test
    )

    # Verify that combined_df has per-iteration importance columns and aggregated columns.
    for i in range(2):
        col = f"importance_{i}"
        assert col in combined_df.columns

    assert "importance_mean" in combined_df.columns
    assert "importance_std" in combined_df.columns
    # Check that the final 'importance' column equals the mean.
    np.testing.assert_allclose(combined_df["importance"].values, combined_df["importance_mean"].values)