import itertools
import re
from typing import List, Tuple
import networkx as nx
import numpy as np
import pandas as pd


def _subset_mapping(
    input_data: List[str], mapping: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    return [m for m in mapping if m[0] in input_data]


def _subset_pathways_on_idx(
    pathways: List[Tuple[str, str]], mapping: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Recursively find all source nodes that are connected to the mapping
    targets. Only pathways whose 'source' is in the resulting set are kept.
    """

    def add_pathways(idx_list, target):
        if not target:
            return idx_list
        updated_idx_list = idx_list + target
        subsetted = [p for p in pathways if p[0] in target]
        new_target = list({p[1] for p in subsetted})
        return add_pathways(updated_idx_list, new_target)

    original_target = list({m[1] for m in mapping})
    idx_list = add_pathways([], original_target)
    return [p for p in pathways if p[0] in idx_list]


def _get_mapping_to_all_layers(
    pathways: List[Tuple[str, str]], mapping: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Build a DataFrame of (input, connections) by traversing the directed graph
    from each mapped translation node.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(pathways)

    components = {"input": [], "connections": []}
    unique_inputs = {m[0] for m in mapping}

    for inp in unique_inputs:
        # Collect all translation nodes for this input
        translation_nodes = [m[1] for m in mapping if m[0] == inp]

        for node_id in translation_nodes:
            if graph.has_node(node_id):
                reachable_nodes = nx.single_source_shortest_path(graph, node_id).keys()
                # Record the connections
                for connection in reachable_nodes:
                    components["input"].append(inp)
                    components["connections"].append(connection)

    df_components = pd.DataFrame(components).drop_duplicates()
    return df_components


def _get_map_from_layer(layer_dict: dict) -> pd.DataFrame:
    """
    Convert a dictionary like {pathway: [inputs]} into a binary matrix
    (rows = pathways, columns = inputs).
    """
    pathways = list(layer_dict.keys())

    unique_inputs = set()
    for inputs_list in layer_dict.values():
        unique_inputs.update(inputs_list)

    unique_inputs = list(unique_inputs)

    # Create an empty matrix of 0's
    mat = np.zeros((len(pathways), len(unique_inputs)), dtype=int)

    # Map pathway -> row index
    pathway_to_idx = {p: i for i, p in enumerate(pathways)}
    # Map input -> column index
    input_to_idx = {inp: j for j, inp in enumerate(unique_inputs)}

    # Fill the matrix
    for p, inputs_list in layer_dict.items():
        row_idx = pathway_to_idx[p]
        for inp in inputs_list:
            col_idx = input_to_idx[inp]
            mat[row_idx, col_idx] = 1

    # Build DataFrame
    df = pd.DataFrame(mat, index=pathways, columns=unique_inputs)

    return df.T


def _add_edges(G: nx.DiGraph, node: str, n_layers: int) -> nx.DiGraph:
    """
    Create 'n_layers' copies of a node (node_copy1, node_copy2, ...),
    connecting each copy to the next, and add them to the graph.
    """
    edges = []
    source = node
    for level in range(n_layers):
        target = f"{node}_copy{level + 1}"
        edges.append((source, target))
        source = target

    G.add_edges_from(edges)
    return G


def _complete_network(G: nx.DiGraph, n_layers: int = 4) -> nx.DiGraph:
    """
    Extend the network so that every terminal node has up to 'n_layers' copies,
    ensuring a uniform depth for all terminal nodes.
    """
    sub_graph = nx.ego_graph(G, "output_node", radius=n_layers)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]

    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source="output_node", target=node))
        if distance <= n_layers:
            diff = n_layers - distance + 1
            _add_edges(sub_graph, node, diff)

    return sub_graph


def _get_nodes_at_level(net: nx.DiGraph, distance: int) -> List[str]:
    """
    Get the set of nodes that lie exactly 'distance' steps away from 'output_node'.
    """
    nodes_at_distance = set(nx.ego_graph(net, "output_node", radius=distance))
    if distance >= 1:
        nodes_one_less = set(nx.ego_graph(net, "output_node", radius=distance - 1))
        nodes_at_distance -= nodes_one_less
    return list(nodes_at_distance)


def _get_layers_from_net(net: nx.DiGraph, n_layers: int) -> List[dict]:
    """
    Extract layer information from the graph, from level 0 up to n_layers-1.
    """
    layers = []
    for dist in range(n_layers):
        nodes = _get_nodes_at_level(net, dist)
        layer_map = {}
        for n in nodes:
            base_name = re.sub(r"_copy.*", "", n)
            successors = list(net.successors(n))
            successors_base = [re.sub(r"_copy.*", "", s) for s in successors]
            layer_map[base_name] = successors_base
        layers.append(layer_map)
    return layers


class PathwayNetwork:
    """
    Represents a network of pathways built from an input set, a list of edges,
    and a mapping that links each input to one or more pathways.
    """

    def __init__(
        self,
        input_data: List[str],
        pathways: List[Tuple[str, str]] = None,
        mapping: List[Tuple[str, str]] = None,
    ):
        """
        :param input_data: e.g., [protein1, protein2, protein3]
        :param pathways: e.g., [(path1, path2), (path3, path4)]
        :param mapping: e.g., [(protein1, path1), (protein2, path3)]
        """
        self.input_data = input_data
        self.pathways = pathways
        self.mapping = mapping 

        # Subset the mapping to keep only entries related to input_data
        self.mapping = _subset_mapping(self.input_data, self.mapping)

        # Subset the pathways by the indices derived from the mapping
        self.pathways = _subset_pathways_on_idx(self.pathways, self.mapping)

        # Expand the mapping to all reachable layers
        self.mapping = _get_mapping_to_all_layers(self.pathways, self.mapping)

        # Maintain a list of unique inputs from the expanded mapping
        self.inputs = sorted(set(self.mapping["input"].to_list()))

        # Build the base network graph
        self.pathway_graph = self.build_network()

    def build_network(self) -> nx.DiGraph:
        """
        Build a directed graph from pathways. Also adds an artificial node,
        'output_node', which connects to all nodes that have no incoming edges.
        """
        pathway_graph = nx.DiGraph()
        pathway_graph.add_edges_from(self.pathways)
        pathway_graph = pathway_graph.reverse()

        # Create a special output node for all sources with no incoming edges
        output_node = "output_node"
        sink_nodes = [n for n, d in pathway_graph.in_degree() if d == 0]
        edges_to_output = [(output_node, node) for node in sink_nodes]
        pathway_graph.add_edges_from(edges_to_output)

        return pathway_graph

    def get_layers(self, n_layers: int) -> List[dict]:
        """
        Get layered structure of the network up to n_layers, plus a final dict
        mapping terminal pathways -> which inputs map to them.

        :param n_layers: Number of layers to extract
        :return: A list of length n_layers + 1:
                 - n_layers layers describing pathway relationships
                 - A final dictionary mapping terminal pathways to input(s).
        """
        completed_graph = _complete_network(self.pathway_graph, n_layers=n_layers)
        layers = _get_layers_from_net(completed_graph, n_layers)

        # Identify terminal nodes in the extended graph
        terminal_nodes = [n for n, d in completed_graph.out_degree() if d == 0]

        # Create a final mapping of terminal pathways -> input(s)
        mapping_df = self.mapping
        terminal_mapping = {}
        missing_pathways = []
        for term_node in terminal_nodes:
            pathway_name = re.sub(r"_copy.*", "", term_node)
            inputs_for_pathway = (
                mapping_df.loc[mapping_df["connections"] == pathway_name, "input"]
                .unique()
                .tolist()
            )

            if not inputs_for_pathway:
                missing_pathways.append(pathway_name)
            terminal_mapping[pathway_name] = inputs_for_pathway

        layers.append(terminal_mapping)
        return layers

    def get_connectivity_matrices(self, n_layers: int) -> List[pd.DataFrame]:
        """
        Produce one connectivity matrix per layer (bottom-up). Each matrix
        displays which inputs are connected to which pathways in that layer.

        :param n_layers: Number of layers
        :return: A list of pandas DataFrames, one per layer
        """
        matrices = []
        layers = self.get_layers(n_layers)

        # Process layers in reverse order so the bottom-most is first
        current_inputs = self.inputs  # will be updated as we go
        for i, layer_dict in enumerate(layers[::-1]):
            layer_matrix = _get_map_from_layer(layer_dict)

            # If this is the bottom-most layer (i=0 in reversed order),
            # update 'self.inputs' with the row index for consistent naming
            if i == 0:
                current_inputs = sorted(layer_matrix.index)
                self.inputs = current_inputs

            # Merge an empty DataFrame (with the old 'current_inputs') to ensure
            # consistent row indices
            placeholder_df = pd.DataFrame(index=current_inputs)
            merged_df = placeholder_df.merge(
                layer_matrix, right_index=True, left_index=True, how="inner"
            )

            # Sort rows and columns for consistency
            merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)
            merged_df = merged_df.sort_index()

            # Prepare for the next iteration
            current_inputs = list(layer_matrix.columns)
            matrices.append(merged_df)

        return matrices


def dataframes_to_pathway_network(
    data_matrix: pd.DataFrame,
    pathway_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    entity_col: str = "Protein",
    source_col: str = "source",
    target_col: str = "target",
    input_col: str = "input",
    translation_col: str = "translation",
) -> PathwayNetwork:
    """
    Construct a PathwayNetwork from three DataFrames:
      1) A 'data_matrix' containing the input entities (e.g. proteins).
      2) A 'pathway_df' containing pathway edges (source -> target).
      3) A 'mapping_df' mapping each entity in data_matrix to pathways.

    This method should be as flexible as possible.

    Args:
        data_matrix (pd.DataFrame): A DataFrame that contains the column 'entity_col'.
          For example, if `entity_col='Protein'`, this DataFrame might have a
          "Protein" column listing protein identifiers.
        pathway_df (pd.DataFrame): A DataFrame with at least two columns describing
          directed edges in a pathway (defaults: 'source', 'target').
        mapping_df (pd.DataFrame): A DataFrame with columns describing how
          each entity maps to pathways (defaults: 'input', 'translation').
        entity_col (str): Name of the column in `data_matrix` containing
          the entities of interest (defaults to 'Protein').
        source_col (str): Name of the column in `pathway_df` representing
          the source node in an edge (defaults to 'source').
        target_col (str): Name of the column in `pathway_df` representing
          the target node in an edge (defaults to 'target').
        input_col (str): Name of the column in `mapping_df` representing
          the input entity (defaults to 'input').
        translation_col (str): Name of the column in `mapping_df` representing
          the mapped pathway node (defaults to 'translation').

    Returns:
        PathwayNetwork: A newly constructed PathwayNetwork instance, ready for
        further analysis (e.g., layer extraction, connectivity matrices).
    """

    input_data = data_matrix[entity_col].dropna().unique().tolist()
    pathways = list(pathway_df[[source_col, target_col]].itertuples(index=False, name=None))
    mapping = list(mapping_df[[input_col, translation_col]].itertuples(index=False, name=None))

    pn = PathwayNetwork(
        input_data=input_data,
        pathways=pathways,
        mapping=mapping,
    )
    return pn
