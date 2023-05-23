import pytest
import pandas as pd
from binn import Network
import numpy as np


def test_build_network():
    # Create sample data for testing
    input_data = pd.DataFrame({"Protein": ["a", "b", "c"]})
    pathways = pd.DataFrame(
        {"child": ["A", "B", "C"], "parent": ["path1", "path1", "path2"]}
    )
    translation = pd.DataFrame(
        {"input": ["a", "b", "c"], "translation": ["A", "B", "C"]}
    )
    # Initialize the Network object
    network = Network(input_data=input_data, pathways=pathways, mapping=translation)

    # Test if the network is built correctly
    expected_nodes = ["root", "A", "B", "C", "path1", "path2"]
    expected_edges = [
        ("path1", "A"),
        ("path1", "B"),
        ("path2", "C"),
        ("root", "path1"),
        ("root", "path2"),
    ]
    assert sorted(network.netx.nodes) == sorted(expected_nodes)
    assert sorted(network.netx.edges) == sorted(expected_edges)


def test_get_layers():
    # Create sample data for testing
    input_data = pd.DataFrame({"Protein": ["a", "b", "c"]})
    pathways = pd.DataFrame(
        {"child": ["A", "B", "C"], "parent": ["path1", "path1", "path2"]}
    )
    translation = pd.DataFrame(
        {"input": ["a", "b", "c"], "translation": ["A", "B", "C"]}
    )
    # Initialize the Network object
    network = Network(input_data=input_data, pathways=pathways, mapping=translation)

    # Test if the layers are returned correctly
    expected_layers = [
        {"root": ["path1", "path2"]},
        {"path1": ["A", "B"], "path2": ["C"]},
        {"A": ["A"], "C": ["C"], "B": ["B"]},
        {
            "A": ["a"],
            "B": ["b"],
            "C": ["c"],
        },
    ]

    assert network.get_layers(3) == expected_layers


def test_get_connectivity_matrices():
    # Create sample data for testing
    input_data = pd.DataFrame({"Protein": ["a", "b", "c"]})
    pathways = pd.DataFrame(
        {"child": ["A", "B", "C"], "parent": ["path1", "path1", "path2"]}
    )
    translation = pd.DataFrame(
        {"input": ["a", "b", "c"], "translation": ["A", "B", "C"]}
    )
    # Initialize the Network object
    network = Network(input_data=input_data, pathways=pathways, mapping=translation)
    print(network.get_connectivity_matrices(2))
    print("\n")
    # Test if the connectivity matrices are returned correctly
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
    matrices = network.get_connectivity_matrices(2)
    for ix in range(len(matrices)):
        assert matrices[ix].equals(expected_matrices[ix])
