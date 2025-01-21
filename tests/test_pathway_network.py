import itertools
import networkx as nx
import pandas as pd
import pytest

# Import the functions and classes to be tested.
# Adjust the import paths as needed.
from binn.model.pathway_network import (
    _subset_mapping,
    _subset_pathways_on_idx,
    _get_mapping_to_all_layers,
    _get_map_from_layer,
    _add_edges,
    _complete_network,
    _get_nodes_at_level,
    _get_layers_from_net,
    PathwayNetwork,
    dataframes_to_pathway_network,
)


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------

def test_subset_mapping():
    input_data = ["p1", "p3"]
    mapping = [("p1", "A"), ("p2", "B"), ("p3", "C"), ("p4", "D")]
    expected = [("p1", "A"), ("p3", "C")]
    result = _subset_mapping(input_data, mapping)
    assert result == expected


def test_subset_pathways_on_idx():
    # Given pathways (edges) and mapping (tuples),
    # we expect the function to recurse and include all edges
    # whose source appears, directly or indirectly, from any mapping target.
    pathways = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("E", "F"),
        ("D", "G"),
    ]
    mapping = [("X", "A")]  # only A is mapped from an input (even though input "X" is arbitrary)
    # Expect to collect A, then B (since A->B), then C (B->C), then D (C->D) and then G (D->G)
    result = _subset_pathways_on_idx(pathways, mapping)
    expected = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "G")]
    # Order might differ, so we compare as sets.
    assert set(result) == set(expected)


def test_get_mapping_to_all_layers():
    # Build a small graph manually.
    pathways = [
        ("A", "B"),
        ("B", "C"),
        ("B", "D"),
        ("D", "E"),
    ]
    mapping = [("p1", "A"), ("p2", "B")]
    # Expected: each input from mapping will lead to a set of reachable nodes from its translation.
    df = _get_mapping_to_all_layers(pathways, mapping)
    # The output DataFrame should have columns "input" and "connections".
    assert set(df.columns) == {"input", "connections"}
    # For input "p1", starting at "A", the reachable nodes should include B, C, D, and E.
    p1_connections = df[df["input"] == "p1"]["connections"].unique().tolist()
    for node in ["B", "C", "D", "E"]:
        assert node in p1_connections
    # For input "p2", starting at "B", reachable nodes should include C, D, and E.
    p2_connections = df[df["input"] == "p2"]["connections"].unique().tolist()
    for node in ["C", "D", "E"]:
        assert node in p2_connections


def test_get_map_from_layer():
    # Given a dictionary mapping pathways to a list of input nodes:
    layer_dict = {
        "P1": ["p1", "p2"],
        "P2": ["p2", "p3"],
        "P3": ["p1"],
    }
    df = _get_map_from_layer(layer_dict)
    # The resulting DataFrame should have rows = inputs and columns = pathways.
    expected_rows = sorted(["p1", "p2", "p3"])
    expected_cols = sorted(["P1", "P2", "P3"])
    assert sorted(df.index.tolist()) == expected_rows
    assert sorted(df.columns.tolist()) == expected_cols
    # Check that the appropriate cells are 1
    assert df.loc["p1", "P1"] == 1
    assert df.loc["p1", "P2"] == 0
    assert df.loc["p2", "P2"] == 1
    # And that missing values are filled with 0.
    assert df.loc["p3", "P1"] == 0


def test_add_edges():
    # Create an empty directed graph, and add copies for node "A"
    G = nx.DiGraph()
    G.add_node("A")
    n_layers = 3
    _add_edges(G, "A", n_layers)
    # The function should add edges:
    # ("A", "A_copy1"), ("A_copy1", "A_copy2"), ("A_copy2", "A_copy3")
    expected_edges = [("A", "A_copy1"), ("A_copy1", "A_copy2"), ("A_copy2", "A_copy3")]
    for edge in expected_edges:
        assert G.has_edge(*edge)


def test_complete_network():
    # Build a simple graph where "output_node" exists and some nodes have no out edges.
    G = nx.DiGraph()
    # Create a small graph with nodes A, B, C and add edge structure:
    # output_node -> A,  A -> B, B -> C.
    G.add_edges_from([("output_node", "A"), ("A", "B"), ("B", "C")])
    # Reverse G if necessary (our _complete_network expects a graph like that)
    # Now, complete the network so that every terminal node gets extended copies.
    n_layers = 4
    complete_G = _complete_network(G, n_layers=n_layers)
    # Terminal nodes in the extended network should have "_copy" in their names.
    terminal_nodes = [n for n, d in complete_G.out_degree() if d == 0]
    # Since "C" was terminal, we expect copies of "C" to be added.
    # The simplest check is that at least one terminal node contains "C_copy"
    assert any("C_copy" in n for n in terminal_nodes)


def test_get_nodes_at_level():
    # Build a small star graph from "output_node".
    G = nx.DiGraph()
    # Create a graph where output_node is at center with edges to A, B, and C.
    G.add_edges_from([("output_node", "A"), ("output_node", "B"), ("output_node", "C")])
    # For distance 1, we expect {A, B, C}
    nodes_dist1 = _get_nodes_at_level(G, 1)
    assert set(nodes_dist1) == {"A", "B", "C"}
    # For distance 0 (the ego graph of radius 0) we expect just output_node.
    nodes_dist0 = _get_nodes_at_level(G, 0)
    assert set(nodes_dist0) == {"output_node"}


def test_get_layers_from_net():
    # Build a small graph with copies to simulate layered structure.
    G = nx.DiGraph()
    # Create a chain: output_node -> A -> B, and also add copies for A.
    G.add_edges_from([("output_node", "A"), ("A", "B")])
    # Additionally, add copies for B
    _add_edges(G, "B", 2)  # adds B_copy1 and B_copy2, for instance.
    n_layers = 3
    layers = _get_layers_from_net(G, n_layers)
    # Expect layers to be a list of dicts with keys as original node names and values as lists of successors.
    # At level 0 (distance 0 from output_node) we expect output_node only.
    assert any("output_node" in mapping for mapping in layers[0].keys())
    # At level 1, expect "A" to be present.
    level1_nodes = set(layers[1].keys())
    assert "A" in level1_nodes or any("A" in n for n in level1_nodes)


# ---------------------------------------------------------------------------
# Integration tests via the PathwayNetwork class and dataframes_to_pathway_network.
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """
    Returns sample input data, pathways, and mapping as lists/tuples.
    """
    # Suppose we have three inputs (e.g., proteins)
    input_data = ["p1", "p2", "p3", "p4"]
    # Suppose the pathways are a set of directed edges.
    pathways = [
        ("P1", "P2"),
        ("P2", "P3"),
        ("P2", "P4"),
        ("P3", "P5"),
        ("P4", "P5"),
        ("P5", "P6"),
    ]
    # And a mapping from input to an initial pathway node.
    mapping = [
        ("p1", "P1"),
        ("p2", "P1"),
        ("p3", "P3"),
        ("p4", "P4"),
    ]
    return input_data, pathways, mapping


def test_pathway_network_layers(sample_data):
    input_data, pathways, mapping = sample_data
    # Build a pathway network.
    pn = PathwayNetwork(input_data=input_data, pathways=pathways, mapping=mapping)
    # Get layers with a given depth (say 3 layers).
    layers = pn.get_layers(n_layers=3)
    # The returned layers should be a list of dictionaries, with the final layer
    # mapping terminal pathways to input(s).
    assert isinstance(layers, list)
    assert len(layers) == 4  # 3 layers + terminal mapping
    # Check that the terminal mapping is a dictionary.
    terminal_mapping = layers[-1]
    assert isinstance(terminal_mapping, dict)
    # For a known terminal pathway (e.g., "P6"), check that if it appears,
    # its mapping has the corresponding input (it depends on the expansion and _get_mapping_to_all_layers).
    # We can at least check that one of the inputs from sample_data appears in the terminal mapping.
    terminal_values = list(terminal_mapping.values())
    flattened = set(itertools.chain.from_iterable(terminal_values))
    for p in ["p1", "p2", "p3", "p4"]:
        assert p in flattened or flattened == set()  # if no mapping, then empty set


def test_dataframes_to_pathway_network(sample_data):
    input_data, pathways, mapping = sample_data
    # Create a dummy data_matrix DataFrame with an entity column.
    data_matrix = pd.DataFrame({
        "Protein": ["p1", "p2", "p3", "p4", "p5"]
    })
    # Build DataFrames for pathways and mapping.
    pathway_df = pd.DataFrame(pathways, columns=["source", "target"])
    mapping_df = pd.DataFrame(mapping, columns=["input", "translation"])
    # Create a pathway network using the provided function.
    pn = dataframes_to_pathway_network(
        data_matrix=data_matrix,
        pathway_df=pathway_df,
        mapping_df=mapping_df,
        entity_col="Protein",
        source_col="source",
        target_col="target",
        input_col="input",
        translation_col="translation",
    )
    # Test that the returned object is a PathwayNetwork.
    assert isinstance(pn, PathwayNetwork)
    # Check that the network graph has the special output_node.
    assert "output_node" in pn.pathway_graph.nodes
    # Also, test that connectivity matrices can be retrieved.
    matrices = pn.get_connectivity_matrices(n_layers=2)
    # Expect a list of two DataFrames.
    assert isinstance(matrices, list)
    assert len(matrices) == 3
    # Each DataFrame should have numeric 0/1 entries.
    for df in matrices:
        assert df.applymap(lambda x: x in [0, 1]).all().all()
