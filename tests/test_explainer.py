import pytest
from binn import BINNExplainer, BINN, Network
import torch
import pandas as pd


@pytest.fixture
def binn_explainer(return_input=False):
    input_data = pd.DataFrame({"Protein": ["a", "b", "c"]})
    pathways = pd.DataFrame(
        {"source": ["A", "B", "C"], "target": ["path1", "path1", "path2"]}
    )
    translation = pd.DataFrame(
        {"input": ["a", "b", "c"], "translation": ["A", "B", "C"]}
    )
    network = Network(input_data=input_data, pathways=pathways, mapping=translation)
    model = BINN(n_layers=2, network=network)
    return BINNExplainer(model)


def test_binn_explainer_init(binn_explainer: BINNExplainer):
    assert isinstance(binn_explainer.model, BINN)

def test_binn_explainer_update_model(binn_explainer: BINNExplainer):
    input_data = pd.DataFrame({"Protein": ["a", "b", "c"]})
    pathways = pd.DataFrame(
        {"source": ["A", "B", "C"], "target": ["path1", "path1", "path2"]}
    )
    translation = pd.DataFrame(
        {"input": ["a", "b", "c"], "translation": ["A", "B", "C"]}
    )
    network = Network(input_data=input_data, pathways=pathways, mapping=translation)
    new_model = BINN(n_layers=2, network=network)
    binn_explainer.update_model(new_model)
    assert binn_explainer.model == new_model


def test_binn_explainer_explain(binn_explainer: BINNExplainer):
    test_data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    background_data = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

    result = binn_explainer.explain(test_data, background_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    
def test_binn_explainer_explain_output(binn_explainer: BINNExplainer):
    test_data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    background_data = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

    result = binn_explainer.explain(test_data, background_data)

    unique_pairs = result[["source name", "target name"]].drop_duplicates()
    unique_pairs = [(s, t) for s, t in zip(unique_pairs["source name"], unique_pairs["target name"])]
    
    theoretical_unique_pairs = []
    for cm in binn_explainer.model.connectivity_matrices:
        for i in cm.index.tolist():
            for j in cm.columns.tolist():
                if cm.loc[i, j] == 1:
                    theoretical_unique_pairs.append((i, j))
                    
    print(unique_pairs)
         
    assert set(unique_pairs) == set(theoretical_unique_pairs)


def test_binn_explainer_explain_input(binn_explainer: BINNExplainer):
    test_data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    background_data = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

    result = binn_explainer.explain_input(test_data, background_data)
    assert isinstance(result, dict)


def test_binn_explainer_explain_layers(binn_explainer: BINNExplainer):
    test_data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    background_data = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
    result = binn_explainer._explain_layers(background_data, test_data)
    assert isinstance(result, dict)
    assert "features" in result
    assert "shap_values" in result


def test_binn_explainer_explain_layer(binn_explainer: BINNExplainer):
    test_data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    background_data = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
    wanted_layer = 0

    result = binn_explainer._explain_layer(background_data, test_data, wanted_layer)
    assert isinstance(result, dict)
    assert "features" in result
    assert "shap_values" in result