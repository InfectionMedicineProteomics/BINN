import itertools
import re
from typing import Union
from binn.plot import shap_sankey
import networkx as nx
import numpy as np
import pandas as pd

"""
Original file from https://github.com/marakeby/pnet_prostate_paper/blob/master/data/pathways/reactome.py
Modified to fit our repo.

"""


class Network:
    """
    A class for building and analyzing a directed graph network of biological pathways.

    Args:
        input_data (pandas.DataFrame): A DataFrame containing the input data for the pathways.
        pathways (pandas.DataFrame): A DataFrame containing information on the pathways.
        mapping (pandas.DataFrame or None, optional): A DataFrame containing mapping information.
            If None, then a DataFrame will be constructed from the `input_data` argument.
            Default is None.
        input_data_column (str, optional): The name of the column in `input_data` that contains
            the input data. Default is 'Protein'.
        subset_pathways (bool, optional): Whether to subset the pathways DataFrame to include
            only those pathways that are relevant to the input data. Default is True.

    Attributes:
        mapping (pandas.DataFrame): A DataFrame containing the mapping information.
        pathways (pandas.DataFrame): A DataFrame containing information on the pathways.
        input_data (pandas.DataFrame): A DataFrame containing the input data for the pathways.
        inputs (list): A list of the unique inputs in the mapping DataFrame.
        netx (networkx.DiGraph): A directed graph network of the pathways.


    """

    def __init__(
        self,
        input_data: pd.DataFrame,
        pathways: pd.DataFrame,
        mapping: Union[pd.DataFrame, None] = None,
        input_data_column: str = "Protein",
        subset_pathways: bool = True
    ):

        if isinstance(mapping, pd.DataFrame):

            self.mapping = mapping

        else:

            self.mapping = pd.DataFrame(
                {
                    "input": input_data[input_data_column].values,
                    "translation": input_data[input_data_column].values,
                }
            )

        if subset_pathways:

            self.mapping = subset_input(
                input_data, self.mapping, input_data_column
            )

            self.pathways = subset_pathways_on_idx(
                pathways, self.mapping)

        else:

            self.pathways = pathways

        self.mapping = _get_mapping_to_all_layers(self.pathways, self.mapping)

        self.input_data = input_data

        self.inputs = self.mapping["input"].unique()

        self.netx = self.build_network()

    def get_terminals(self) -> list:
        """
        Returns a list of all terminal nodes (nodes with no outgoing edges) in the network.

        Returns:
            A list of strings representing the terminal nodes in the network.
        """

        return [n for n, d in self.netx.out_degree() if d == 0]

    def get_roots(self):
        """
        Returns a list of all root nodes (nodes with no incoming edges) in the network.

        Returns:
            A list of strings representing the root nodes in the network.
        """
        return get_nodes_at_level(self.netx, distance=1)

    def build_network(self):
        """
        Constructs a networkx DiGraph from the edges in the 'pathways' attribute of the object, with a root node added to the graph to connect all root nodes together.

        Returns:
            A networkx DiGraph object representing the constructed network.
        """
        if hasattr(self, "netx"):
            return self.netx

        net = nx.from_pandas_edgelist(
            self.pathways, "parent", "child", create_using=nx.DiGraph()
        )
        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = "root"
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def get_tree(self):
        """
        Returns a BFS tree of the network from the root node.

        Returns:
            A networkx DiGraph object representing the BFS tree of the network from the root node.
        """
        return nx.bfs_tree(
            self.netx, "root"
        )

    def get_completed_network(self, n_levels):
        """
        Returns a network that has been completed up to a certain number of levels below the root node.

        Args:
            n_levels: The number of levels below the root node to complete the network to.

        Returns:
            A networkx DiGraph object representing the completed network.
        """
        return complete_network(self.netx, n_levels=n_levels)

    def get_completed_tree(self, n_levels):
        """
        Returns a BFS tree of the completed network up to a certain number of levels below the root node.

        Args:
            n_levels: The number of levels below the root node to complete the network to.

        Returns:
            A networkx DiGraph object representing the completed BFS tree of the network from the root node.
        """
        G = self.get_tree()
        return complete_network(G, n_levels=n_levels)

    def get_layers(self, n_levels, direction="root_to_leaf") -> list:
        """
        Returns a list of dictionaries where each dictionary contains the pathways at a certain level of the completed network and their inputs.

        Args:
            n_levels: The number of levels below the root node to complete the network to.
            direction: The direction of the layers to return. Must be either "root_to_leaf" or "leaf_to_root". Defaults to "root_to_leaf".

        Returns:
            A list of dictionaries, where each dictionary contains pathway names as keys and input lists as values.
        """
        if direction == "root_to_leaf":
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)

        terminal_nodes = [
            n for n, d in net.out_degree() if d == 0
        ]

        mapping_df = self.mapping
        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub("_copy.*", "", p)
            inputs = mapping_df[mapping_df["connections"] == pathway_name][
                "input"
            ].unique()
            if len(inputs) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = inputs
        layers.append(dict)
        return layers

    def get_connectivity_matrices(self, n_levels, direction="root_to_leaf") -> list:
        """
        Returns a list of connectivity matrices for each layer of the completed network, ordered from leaf nodes to root node.

        Args:
            n_levels: The number of levels below the root node to complete the network to.
            direction: The direction of the layers to return. Must be either "root_to_leaf" or "leaf_to_root". Defaults to "root_to_leaf".

        Returns:
            A list of pandas DataFrames representing the connectivity matrices for each layer of the completed network.
        """
        connectivity_matrices = []
        layers = self.get_layers(n_levels, direction)
        for i, layer in enumerate(layers[::-1]):
            mapp = get_map_from_layer(layer)
            if i == 0:
                inputs = list(mapp.index)
                self.inputs = inputs
            filter_df = pd.DataFrame(index=inputs)
            all = filter_df.merge(mapp, right_index=True,
                                  left_index=True, how="inner")
            inputs = list(mapp.columns)
            connectivity_matrices.append(all)
        return connectivity_matrices


def get_map_from_layer(layer_dict):
    pathways = layer_dict.keys()
    inputs = list(itertools.chain.from_iterable(layer_dict.values()))
    inputs = list(np.unique(inputs))
    df = pd.DataFrame(index=pathways, columns=inputs)
    for k, v in layer_dict.items():
        df.loc[k, v] = 1
    df = df.fillna(0)
    return df.T


def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + "_copy" + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def complete_network(G, n_levels=4):
    nr_copies = 0
    sub_graph = nx.ego_graph(G, "root", radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source="root", target=node))
        if distance <= n_levels:
            nr_copies = nr_copies + n_levels - distance
            diff = n_levels - distance + 1
            sub_graph = add_edges(
                sub_graph, node, diff
            )
    return sub_graph


def get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, "root", radius=distance))
    if distance >= 1.0:
        nodes -= set(nx.ego_graph(net, "root", radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub(
                "_copy.*", "", n
            )
            next = net.successors(n)
            dict[n_name] = [
                re.sub("_copy.*", "", nex) for nex in next
            ]
        layers.append(
            dict
        )
    return layers


def get_separation(pathways, input_data, translation_mapping):
    sep_path = ","
    sep_input = ","
    sep_transl = ","
    if pathways.endswith("tsv"):
        sep_path = "\t"
    if input_data.endswith("tsv"):
        sep_input = "\t"
    if translation_mapping.endswith("tsv"):
        sep_transl = "\t"
    return sep_path, sep_input, sep_transl


def subset_input(
    input_df, translation, input_data_column,
):
    keys_in_data = input_df[input_data_column].unique()
    translation = translation[translation["input"].isin(keys_in_data)]
    return translation


def subset_pathways_on_idx(pathways, translation):
    def add_pathways(counter, idx_list, parent):
        counter += 1
        if len(parent) == 0:

            return idx_list
        else:
            idx_list = idx_list + parent
            subsetted_pathway = pathways[pathways["child"].isin(parent)]
            new_parent = list(subsetted_pathway["parent"].unique())
            return add_pathways(counter, idx_list, new_parent)

    counter = 0
    original_parent = list(translation["translation"].unique())
    idx_list = []
    idx_list = add_pathways(counter, idx_list, original_parent)
    pathways = pathways[pathways["child"].isin(idx_list)]
    return pathways


class ImportanceNetwork:
    """
    Class for generating and analyzing importance networks.

    This class generates and analyzes directed graphs representing importance networks, in which nodes represent variables and edges represent dependencies or influences between the variables. The graphs are constructed from a pandas DataFrame specifying the relationships between the variables, with columns for the source and target nodes, the weight of the edge, and optionally the layers of the source and target nodes. The graphs can be used to calculate various properties of the importance network, such as the number of nodes in upstream or downstream subgraphs, the fan-in and fan-out of nodes, and the weighted values of nodes. The class also includes a method for generating a Sankey diagram of the importance network.

    Parameters:
        df (pandas.DataFrame): A DataFrame specifying the relationships between the variables, with columns for the source and target nodes, the weight of the edge, and optionally the layers of the source and target nodes.
        val_col (str, optional): The name of the column in df containing the weight of the edge. Default is "value".

    """

    def __init__(self, df: pd.DataFrame, val_col: str = "value"):
        self.df = df
        self.val_col = val_col
        self.G = self.create_graph()
        self.G_reverse = self.G.reverse()

    def create_graph(self):
        """
        Create a directed graph (DiGraph) from the source and target nodes in the input dataframe.

        Returns:
            G: a directed graph (DiGraph) object
        """
        self.df["source"] = self.df["source"].apply(lambda x: x.split("_")[0])
        self.df["target"] = self.df["target"].apply(lambda x: x.split("_")[0])
        G = nx.DiGraph()
        for k in self.df.iterrows():
            source = k[1]["source"]
            value = k[1][self.val_col]
            source_layer = k[1]["source layer"] + 1
            G.add_node(source, weight=value, layer=source_layer)
        for k in self.df.iterrows():
            source = k[1]["source"]
            target = k[1]["target"]
            G.add_edge(source, target)
        root_layer = max(self.df["target layer"]) + 1
        G.add_node("root", weight=0, layer=root_layer)
        return G

    def get_downstream_subgraph(self, query_node: str, depth_limit=None):
        """
        Get a subgraph that contains all nodes downstream of the query node up to the given depth limit (if provided).

        Args:
            query_node: a string representing the name of the node from which the downstream subgraph is constructed
            depth_limit: an integer representing the maximum depth to which the subgraph is constructed (optional)

        Returns:
            SG: a directed graph (DiGraph) object containing all nodes downstream of the query node, up to the given depth limit
        """
        SG = nx.DiGraph()
        nodes = [
            n
            for n in nx.traversal.bfs_successors(
                self.G, query_node, depth_limit=depth_limit
            )
            if n != query_node
        ]
        for source, targets in nodes:
            SG.add_node(source, **self.G.nodes()[source])
            for t in targets:
                SG.add_node(t, **self.G.nodes()[t])
        for node1 in SG.nodes():
            for node2 in SG.nodes():
                if self.G.has_edge(node1, node2):
                    SG.add_edge(node1, node2)

        SG.add_node(query_node, **self.G.nodes()[query_node])
        return SG

    def get_upstream_subgraph(self, query_node: str, depth_limit=None):
        """
        Get a subgraph that contains all nodes upstream of the query node up to the given depth limit (if provided).

        Args:
            query_node: a string representing the name of the node from which the upstream subgraph is constructed
            depth_limit: an integer representing the maximum depth to which the subgraph is constructed (optional)

        Returns:
            SG: a directed graph (DiGraph) object containing all nodes upstream of the query node, up to the given depth limit
        """
        SG = self.get_downstream_subgraph(query_node, depth_limit=depth_limit)
        return SG

    def get_complete_subgraph(self, query_node: str, depth_limit=None):
        """
        Get a subgraph that contains all nodes both upstream and downstream of the query node up to the given depth limit (if provided).

        Args:
            query_node: a string representing the name of the node from which the complete subgraph is constructed
            depth_limit: an integer representing the maximum depth to which the subgraph is constructed (optional)

        Returns:
            SG: a directed graph (DiGraph) object containing all nodes both upstream and downstream of the query node, up to the given depth limit
        """
        SG = self.get_downstream_subgraph(self.G, query_node)
        nodes = [
            n
            for n in nx.traversal.bfs_successors(
                self.G_reverse, query_node, depth_limit=depth_limit
            )
            if n != query_node
        ]
        for source, targets in nodes:
            SG.add_node(source, **self.G_reverse.nodes()[source])
            for t in targets:
                SG.add_node(t, **self.G_reverse.nodes()[t])
        for node1 in SG.nodes():
            for node2 in SG.nodes():
                if self.G_reverse.has_edge(node1, node2):
                    SG.add_edge(node1, node2)

        SG.add_node(query_node, **self.G_reverse.nodes()[query_node])
        return SG

    def get_nr_nodes_in_upstream_SG(self, query_node: str):
        """
        Get the number of nodes in the upstream subgraph of the query node.

        Args:
            query_node: a string representing the name of the node from which the upstream subgraph is constructed

        Returns:
            the number of nodes in the upstream subgraph of the query node
        """

        SG = self.get_upstream_subgraph(query_node, depth_limit=None)
        return SG.number_of_nodes()

    def get_nr_nodes_in_downstream_SG(self, query_node: str):
        """
        Get the number of nodes in the downstream subgraph of the query node.

        Args:
            query_node: a string representing the name of the node from which the downstream subgraph is constructed

        Returns:
            the number of nodes in the downstream subgraph of the query node
        """
        SG = self.get_downstream_subgraph(query_node, depth_limit=None)
        return SG.number_of_nodes()

    def get_fan_in(self, query_node: str):
        """
        Get the number of incoming edges (fan-in) for the query node.

        Args:
            query_node: a string representing the name of the node

        Returns:
            the number of incoming edges (fan-in) for the query node
        """
        return len([n for n in self.G.in_edges(query_node)])

    def get_fan_out(self, query_node: str):
        """
        Get the number of outgoing edges (fan-out) for the query node.

        Args:
            query_node: a string representing the name of the node

        Returns:
            the number of outgoing edges (fan-out) for the query node
        """
        return len([n for n in self.G.out_edges(query_node)])

    def add_normalization(self):
        """
        Adds several normalization columns to the dataframe:
        - 'fan_in': the number of incoming edges for each node

        - 'fan_out': the number of outgoing edges for each node

        - 'fan_tot': the total number of incoming and outgoing edges for each node

        - 'nodes_in_upstream': the number of nodes in the upstream subgraph for each node

        - 'nodes_in_downstream': the number of nodes in the downstream subgraph for each node

        - 'nodes_in_SG': the total number of nodes in the upstream and downstream subgraphs for each node

        - 'log(nodes_in_SG)': the logarithm (base 2) of 'nodes_in_SG'

        - 'weighted_val_log': the 'value' column normalized by the logarithm of 'nodes_in_SG'

        Returns:
            pd.DataFrame:
                The input dataframe with the added normalization columns.
        """
        self.df["fan_in"] = self.df.apply(
            lambda x: self.get_fan_in(x["source"]), axis=1
        )
        self.df["fan_out"] = self.df.apply(
            lambda x: self.get_fan_out(x["source"]), axis=1
        )
        self.df["fan_tot"] = self.df.apply(
            lambda x: x["fan_in"] + x["fan_out"], axis=1)
        self.df["nodes_in_upstream"] = self.df.apply(
            lambda x: self.get_nr_nodes_in_upstream_SG(x["source"]), axis=1
        )
        self.df["nodes_in_downstream"] = self.df.apply(
            lambda x: self.get_nr_nodes_in_downstream_SG(x["source"]), axis=1
        )
        self.df["nodes_in_SG"] = (
            self.df["nodes_in_downstream"] + self.df["nodes_in_upstream"]
        )
        self.df["log(nodes_in_SG)"] = np.log2(self.df["nodes_in_SG"])
        self.df["weighted_val_log"] = self.df.apply(
            lambda x: x["value"] / (np.log2(x["nodes_in_SG"])), axis=1
        )
        return self.df

    def generate_sankey(
        self,
        query_node: str,
        upstream: bool = False,
        savename: str = "sankey.png",
        val_col: str = "value",
        cmap_name: str = "coolwarm",
    ):
        """
        Generate a Sankey diagram using the provided query node.

        Args:
            query_node (str): The node to use as the starting point for the Sankey diagram.
            upstream (bool, optional): If True, the Sankey diagram will show the upstream flow of the
                query_node. If False (default), the Sankey diagram will show the downstream flow of the
                query_node.
            savename (str, optional): The file name to save the Sankey diagram as. Defaults to "sankey.png".
            val_col (str, optional): The column in the DataFrame that represents the value flow between
                nodes. Defaults to "value".
            cmap_name (str, optional): The name of the color map to use for the Sankey diagram. Defaults
                to "coolwarm".

        Returns:
            None: The Sankey diagram will be saved as an image file.

        Raises:
            ValueError: If the provided query_node is not present in the graph.

        """
        if upstream == False:
            final_node = "root"
            SG = self.get_downstream_subgraph(query_node, depth_limit=None)
            source_or_target = "source"
        else:
            final_node = query_node
            SG = self.get_upstream_subgraph(query_node, depth_limit=None)
            source_or_target = "target"
        nodes_in_SG = [n for n in SG.nodes]
        df = self.df[self.df[source_or_target].isin(nodes_in_SG)]
        fig = shap_sankey(
            df, final_node=final_node, val_col=val_col, cmap_name=cmap_name
        )

        fig.write_image(f"{savename}", width=1200, scale=2.5, height=500)


def _get_mapping_to_all_layers(pathways, mapping):
    graph = nx.from_pandas_edgelist(
        pathways, source="child", target="parent", create_using=nx.DiGraph()
    )
    components = {"input": [], "connections": []}
    for translation in mapping["input"]:
        ids = mapping[mapping["input"]
                      == translation]["translation"]
        for id in ids:

            connections = graph.subgraph(
                nx.single_source_shortest_path(graph, id).keys()
            ).nodes
            for connection in connections:
                components["input"].append(translation)
                components["connections"].append(connection)
    components = pd.DataFrame(components)
    components.drop_duplicates(inplace=True)
    return components
