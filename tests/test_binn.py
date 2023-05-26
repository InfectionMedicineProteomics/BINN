import pytest
from binn import BINN, Network
import pandas as pd
import torch


@pytest.fixture
def binn_model():
    input_data = pd.DataFrame({"Protein": ["a", "b", "c"]})
    pathways = pd.DataFrame(
        {"child": ["A", "B", "C"], "parent": ["path1", "path1", "path2"]}
    )
    translation = pd.DataFrame(
        {"input": ["a", "b", "c"], "translation": ["A", "B", "C"]}
    )
    network = Network(input_data=input_data, pathways=pathways, mapping=translation)
    model = BINN(n_layers=2, network=network)
    return model


def test_forward_pass(binn_model):
    x = torch.randn(32, 3)  # input tensor with shape (batch_size, input_size)
    output = binn_model.forward(x)
    assert output.shape == (32, 2)  # output shape should match (batch_size, n_outputs)


def test_training_step(binn_model):
    batch = (torch.randn(32, 3), torch.randint(2, (32,)))  # example batch of data
    loss = binn_model.training_step(batch, 1)
    assert isinstance(loss, torch.Tensor)


def test_validation_step(binn_model):
    batch = (torch.randn(32, 3), torch.randint(2, (32,)))  # example batch of data
    result = binn_model.validation_step(batch, 1)
    assert isinstance(result, dict)
    assert "val_loss" in result
    assert "val_acc" in result


def test_test_step(binn_model):
    batch = (torch.randn(32, 3), torch.randint(2, (32,)))  # example batch of data
    binn_model.test_step(batch, 1)


def test_configure_optimizers(binn_model):
    optimizers, schedulers = binn_model.configure_optimizers()
    assert isinstance(optimizers, list)
    assert isinstance(schedulers, list)
    assert len(optimizers) == 1
    assert len(schedulers) == 1


def test_calculate_accuracy(binn_model):
    y = torch.tensor([0, 1, 1, 0])
    prediction = torch.tensor([0, 1, 0, 1])
    accuracy = binn_model.calculate_accuracy(y, prediction)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_get_connectivity_matrices(binn_model):
    matrices = binn_model.get_connectivity_matrices()
    expected_matrices = [
        pd.DataFrame(
            data=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            columns=["A", "B", "C"],
            index=["a", "b", "c"],
        ),
        pd.DataFrame(
            data=[[1, 0], [1, 0], [0, 1]],
            columns=["path1", "path2"],
            index=["A", "B", "C"],
        ),
        pd.DataFrame(
            data=[[1], [1]],
            columns=["root"],
            index=["path1", "path2"],
        ),
    ]
    assert isinstance(matrices, list)
    for ix in range(len(matrices)):
        assert matrices[ix].equals(expected_matrices[ix])
