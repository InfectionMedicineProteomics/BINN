import pytest
from binn import BINNExplainer, BINN, Network
import torch
import pandas as pd


@pytest.fixture
def binn_explainer():
    input_data = pd.DataFrame({"Protein": ["a", "b", "c"]})
    pathways = pd.DataFrame(
        {"child": ["A", "B", "C"], "parent": ["path1", "path1", "path2"]}
    )
    translation = pd.DataFrame(
        {"input": ["a", "b", "c"], "translation": ["A", "B", "C"]}
    )
    network = Network(input_data=input_data, pathways=pathways, mapping=translation)
    model = BINN(n_layers=2, network=network)

    return BINNExplainer(model)


def test_binn_explainer_init(binn_explainer):
    assert isinstance(binn_explainer.model, BINN)


def test_binn_explainer_update_model(binn_explainer):
    input_data = pd.DataFrame({"Protein": ["a", "b", "c"]})
    pathways = pd.DataFrame(
        {"child": ["A", "B", "C"], "parent": ["path1", "path1", "path2"]}
    )
    translation = pd.DataFrame(
        {"input": ["a", "b", "c"], "translation": ["A", "B", "C"]}
    )
    network = Network(input_data=input_data, pathways=pathways, mapping=translation)
    new_model = BINN(n_layers=2, network=network)
    binn_explainer.update_model(new_model)
    assert binn_explainer.model == new_model


# Test the explain method
def test_binn_explainer_explain(binn_explainer):
    test_data = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    background_data = torch.Tensor([[7, 8, 9], [10, 11, 12]])

    result = binn_explainer.explain(test_data, background_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


# Test the explain_input method
def test_binn_explainer_explain_input(binn_explainer):
    # Create test and background data tensors
    test_data = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    background_data = torch.Tensor([[7, 8, 9], [10, 11, 12]])

    result = binn_explainer.explain_input(test_data, background_data)
    assert isinstance(result, dict)
    # Add assertions for the expected output or behavior
