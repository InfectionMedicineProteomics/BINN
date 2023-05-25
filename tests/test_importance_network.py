import pandas as pd
import pytest
import networkx as nx
from binn import ImportanceNetwork


@pytest.fixture
def importance_df():
    df = pd.DataFrame(
        {
            "source name": ["A", "B", "C", "D", "E", "F"],
            "source": [1, 2, 3, 4, 5, 6],
            "target": [4, 5, 6, 0, 0, 0],
            "value": [10, 10, 10, 20, 20, 20],
            "source layer": [0, 0, 0, 1, 1, 1],
            "target layer": [1, 1, 1, 2, 2, 2],
        }
    )
    return df


def test_create_graph(importance_df):
    importance_network = ImportanceNetwork(importance_df)
    graph = importance_network.create_graph()

    assert isinstance(graph, nx.DiGraph)

    assert len(graph.nodes) == 7
    assert len(graph.edges) == 6


def test_get_downstream_subgraph(importance_df):
    importance_network = ImportanceNetwork(importance_df)
    subgraph = importance_network.get_downstream_subgraph(1)

    assert isinstance(subgraph, nx.DiGraph)
    assert len(subgraph.nodes) == 3
    assert len(subgraph.edges) == 2


def test_get_upstream_subgraph(importance_df):
    network = ImportanceNetwork(importance_df)
    subgraph = network.get_upstream_subgraph(5, depth_limit=None)
    assert isinstance(subgraph, nx.DiGraph)
    assert subgraph.number_of_nodes() == 2
    assert subgraph.number_of_edges() == 1


def test_get_complete_subgraph(importance_df):
    network = ImportanceNetwork(importance_df)
    subgraph = network.get_complete_subgraph(1, depth_limit=None)
    assert isinstance(subgraph, nx.DiGraph)
    assert subgraph.number_of_nodes() == 3
    assert subgraph.number_of_edges() == 3


def test_get_nr_nodes_in_upstream_subgraph(importance_df):
    network = ImportanceNetwork(importance_df)
    nr_nodes = network.get_nr_nodes_in_upstream_subgraph(5)
    assert nr_nodes == 2


def test_get_nr_nodes_in_downstream_subgraph(importance_df):
    network = ImportanceNetwork(importance_df)
    nr_nodes = network.get_nr_nodes_in_downstream_subgraph(5)
    assert nr_nodes == 2


def test_get_fan_in(importance_df):
    network = ImportanceNetwork(importance_df)
    fan_in = network.get_fan_in(5)
    assert fan_in == 1


def test_get_fan_out(importance_df):
    network = ImportanceNetwork(importance_df)
    fan_out = network.get_fan_out(5)
    assert fan_out == 1


def test_add_normalization(importance_df):
    network = ImportanceNetwork(importance_df, normalize=False)
    assert network.importance_df["value"].sum() == 90
    network.add_normalization()
    assert network.importance_df["value"].sum() < 90
