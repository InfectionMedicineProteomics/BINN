from typing import Union
from binn.plot import subgraph_sankey, complete_sankey
import networkx as nx
import numpy as np
import pandas as pd


class ImportanceNetwork:
    """
    A class for building and analyzing a directed graph representing the importance network of a system.

    Parameters:
        importance_df (pandas.DataFrame): A dataframe with columns for the source node, target node, value flow between nodes,
            and layer for each node. This dataframe should represent the complete importance network of the system.
        val_col (str, optional): The name of the column in the DataFrame that represents the value flow between
            nodes. Defaults to "value".
        normalize (bool, optional): Whether or not to normalize the value column. Defaults to True.
        norm_method (str, optional): Method to normalie the value column with. Options are 'subgraph' and 'fan'.
            If 'subgraph', normalizes on the log(nodes in subgraph) from each node. If 'fan', normalizes on the
            log(fan in + fan out) for each node.

    Attributes:
        importance_df (pandas.DataFrame): The dataframe used for downstream/upstream subgraph construction and plotting.
        val_col (str): The name of the column in the DataFrame that represents the value flow between nodes.
        G (networkx.DiGraph): A directed graph object representing the importance network of the system.
        G_reverse (networkx.DiGraph): A directed graph object representing the importance network of the system in reverse.
        norm_method (str): The normalization method.

    """

    def __init__(
        self,
        importance_df: pd.DataFrame,
        normalize: bool = True,
        norm_method: str = "subgraph",
        val_col: str = "value",
    ):
        self.root_node = 0
        self.importance_df = importance_df
        self.val_col = val_col
        self.importance_graph = self.create_graph()
        self.importance_graph_reverse = self.importance_graph.reverse()
        self.norm_method = norm_method
        if normalize:
            self.importance_df = self.add_normalization(method=norm_method)

    def plot_subgraph_sankey(
        self,
        query_node: str,
        upstream: bool = False,
        savename: str = "sankey.png",
        val_col: str = "value",
        cmap: str = "coolwarm",
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
            plotly.graph_objs._figure.Figure: The plotly Figure object representing the Sankey diagram.

        """
        if not upstream:
            final_node = self.get_node("root")

            subgraph = self.get_downstream_subgraph(query_node, depth_limit=None)
            source_or_target = "source"
        else:
            final_node = self.get_node(query_node)
            subgraph = self.get_upstream_subgraph(query_node, depth_limit=None)
            source_or_target = "target"
        nodes_in_subgraph = [n for n in subgraph.nodes]
        df = self.importance_df[
            self.importance_df[source_or_target].isin(nodes_in_subgraph)
        ].copy()
        fig = subgraph_sankey(
            df, final_node=final_node, val_col=val_col, cmap_name=cmap
        )

        fig.write_image(f"{savename}", width=1200, scale=2.5, height=500)
        return fig

    def plot_complete_sankey(
        self,
        multiclass: bool = False,
        show_top_n: int = 10,
        node_cmap: str = "Reds",
        edge_cmap: Union[str, list] = "Reds",
        savename="sankey.png",
    ):
        """
        Plot a complete Sankey diagram for the importance network.

        Parameters:
            multiclass : bool, optional
                If True, plot multiclass Sankey diagram. Defaults to False.
            show_top_n : int, optional
                Show only the top N nodes in the Sankey diagram. Defaults to 10.
            node_cmap : str, optional
                The color map for the nodes. Defaults to "Reds".
            edge_cmap : str or list, optional
                The color map for the edges. Defaults to "Reds".
            savename : str, optional
                The filename to save the plot. Defaults to "sankey.png".

        Returns:
            plotly.graph_objs._figure.Figure: The plotly Figure object representing the Sankey diagram.
        """

        fig = complete_sankey(
            self.importance_df,
            multiclass=multiclass,
            val_col=self.val_col,
            show_top_n=show_top_n,
            edge_cmap=edge_cmap,
            node_cmap=node_cmap,
        )

        fig.write_image(f"{savename}", width=1900, scale=2, height=800)
        return fig

    def create_graph(self):
        """
        Create a directed graph (DiGraph) from the source and target nodes in the input dataframe.

        Returns:
            importance_graph: a directed graph (DiGraph) object
        """
        importance_graph = nx.DiGraph()
        for k in self.importance_df.iterrows():
            source_name = k[1]["source name"]
            source = k[1]["source"]
            value = k[1][self.val_col]
            source_layer = k[1]["source layer"] + 1
            importance_graph.add_node(
                source, weight=value, layer=source_layer, name=source_name
            )
        for k in self.importance_df.iterrows():
            source = k[1]["source"]
            target = k[1]["target"]
            importance_graph.add_edge(source, target)
        root_layer = max(self.importance_df["target layer"]) + 1
        importance_graph.add_node(
            self.root_node, weight=0, layer=root_layer, name="root"
        )
        return importance_graph

    def get_downstream_subgraph(self, query_node: str, depth_limit=None):
        """
        Get a subgraph that contains all nodes downstream of the query node up to the given depth limit (if provided).

        Args:
            query_node: a string representing the name of the node from which the downstream subgraph is constructed
            depth_limit: an integer representing the maximum depth to which the subgraph is constructed (optional)

        Returns:
            subgraph: a directed graph (DiGraph) object containing all nodes downstream of the query node, up to the given depth limit
        """
        subgraph = nx.DiGraph()
        nodes = [
            n
            for n in nx.traversal.bfs_successors(
                self.importance_graph, query_node, depth_limit=depth_limit
            )
            if n != query_node
        ]
        for source, targets in nodes:
            subgraph.add_node(source, **self.importance_graph.nodes()[source])
            for t in targets:
                subgraph.add_node(t, **self.importance_graph.nodes()[t])
        for node1 in subgraph.nodes():
            for node2 in subgraph.nodes():
                if self.importance_graph.has_edge(node1, node2):
                    subgraph.add_edge(node1, node2)

        subgraph.add_node(query_node, **self.importance_graph.nodes()[query_node])
        return subgraph

    def get_upstream_subgraph(self, query_node: str, depth_limit=None):
        """
        Get a subgraph that contains all nodes upstream of the query node up to the given depth limit (if provided).

        Args:
            query_node: a string representing the name of the node from which the upstream subgraph is constructed
            depth_limit: an integer representing the maximum depth to which the subgraph is constructed (optional)

        Returns:
            subgraph: a directed graph (DiGraph) object containing all nodes upstream of the query node, up to the given depth limit
        """
        subgraph = self.get_downstream_subgraph(query_node, depth_limit=depth_limit)
        return subgraph

    def get_complete_subgraph(self, query_node: str, depth_limit=None):
        """
        Get a subgraph that contains all nodes both upstream and downstream of the query node up to the given depth limit (if provided).

        Args:
            query_node: a string representing the name of the node from which the complete subgraph is constructed
            depth_limit: an integer representing the maximum depth to which the subgraph is constructed (optional)

        Returns:
            subgraph: a directed graph (DiGraph) object containing all nodes both upstream and downstream of the query node, up to the given depth limit
        """
        subgraph = self.get_downstream_subgraph(query_node, depth_limit=depth_limit)
        nodes = [
            n
            for n in nx.traversal.bfs_successors(
                self.importance_graph_reverse, query_node, depth_limit=depth_limit
            )
            if n != query_node
        ]
        for source, targets in nodes:
            subgraph.add_node(source, **self.importance_graph_reverse.nodes()[source])
            for t in targets:
                subgraph.add_node(t, **self.importance_graph_reverse.nodes()[t])
        for node1 in subgraph.nodes():
            for node2 in subgraph.nodes():
                if self.importance_graph_reverse.has_edge(node1, node2):
                    subgraph.add_edge(node1, node2)

        subgraph.add_node(
            query_node, **self.importance_graph_reverse.nodes()[query_node]
        )
        return subgraph

    def get_nr_nodes_in_upstream_subgraph(self, query_node: str):
        """
        Get the number of nodes in the upstream subgraph of the query node.

        Args:
            query_node: a string representing the name of the node from which the upstream subgraph is constructed

        Returns:
            the number of nodes in the upstream subgraph of the query node
        """

        subgraph = self.get_upstream_subgraph(query_node, depth_limit=None)
        return subgraph.number_of_nodes()

    def get_nr_nodes_in_downstream_subgraph(self, query_node: str):
        """
        Get the number of nodes in the downstream subgraph of the query node.

        Args:
            query_node: a string representing the name of the node from which the downstream subgraph is constructed

        Returns:
            the number of nodes in the downstream subgraph of the query node
        """
        subgraph = self.get_downstream_subgraph(query_node, depth_limit=None)
        return subgraph.number_of_nodes()

    def get_fan_in(self, query_node: str):
        """
        Get the number of incoming edges (fan-in) for the query node.

        Args:
            query_node: a string representing the name of the node

        Returns:
            the number of incoming edges (fan-in) for the query node
        """
        return len([n for n in self.importance_graph.in_edges(query_node)])

    def get_fan_out(self, query_node: str):
        """
        Get the number of outgoing edges (fan-out) for the query node.

        Args:
            query_node: a string representing the name of the node

        Returns:
            the number of outgoing edges (fan-out) for the query node
        """
        return len([n for n in self.importance_graph.out_edges(query_node)])

    def add_normalization(self, method: str = "subgraph"):
        """
        Adds normalization to the importance values based on the specified method.

        Args:
            method (str): The normalization method to use. Options are "fan" and "subgraph".
                "fan" normalizes based on fan-in and fan-out values.
                "subgraph" normalizes based on the number of nodes in the upstream and
                downstream subgraphs.

        Returns:
            pd.DataFrame: The importance dataframe with the normalized values added.
        """
        if method == "fan":
            fan_in = np.array([self.get_fan_in(x) for x in self.importance["source"]])
            fan_out = np.array([self.get_fan_out(x) for x in self.importance["source"]])
            nr_tot = fan_in + fan_out
        if method == "subgraph":
            upstream_nodes = np.array(
                [
                    self.get_nr_nodes_in_upstream_subgraph(x)
                    for x in self.importance_df["source"]
                ]
            )
            downstream_nodes = np.array(
                [
                    self.get_nr_nodes_in_downstream_subgraph(x)
                    for x in self.importance_df["source"]
                ]
            )
            nr_tot = upstream_nodes + downstream_nodes

        self.importance_df["value"] = self.importance_df["value"] / np.log2(nr_tot)
        return self.importance_df

    def get_node(self, name):
        for node, d in self.importance_graph.nodes(data=True):
            if d["name"] == name:
                return node
        raise ValueError(f"Could not find node {name}")
