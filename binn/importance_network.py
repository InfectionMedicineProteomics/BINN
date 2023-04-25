from typing import Union
import plotly
from binn.plot import subgraph_sankey, complete_sankey
import networkx as nx
import numpy as np
import pandas as pd


class ImportanceNetwork:
    """
    A class for building and analyzing a directed graph representing the importance network of a system.

    Parameters:
        df (pandas.DataFrame): A dataframe with columns for the source node, target node, value flow between nodes,
            and layer for each node. This dataframe should represent the complete importance network of the system.
        val_col (str, optional): The name of the column in the DataFrame that represents the value flow between
            nodes. Defaults to "value".

    Attributes:
        complete_df (pandas.DataFrame): The original dataframe containing the complete importance network of the system.
        df (pandas.DataFrame): The dataframe used for downstream/upstream subgraph construction and plotting.
        val_col (str): The name of the column in the DataFrame that represents the value flow between nodes.
        G (networkx.DiGraph): A directed graph object representing the importance network of the system.
        G_reverse (networkx.DiGraph): A directed graph object representing the importance network of the system in reverse.

    """

    def __init__(self, df: pd.DataFrame, val_col: str = "value"):
        self.complete_df = df
        self.df = df
        self.val_col = val_col
        self.G = self.create_graph()
        self.G_reverse = self.G.reverse()

    def plot_subgraph_sankey(
        self,
        query_node: str,
        upstream: bool = False,
        savename: str = "sankey.png",
        val_col: str = "value",
        cmap: str = "coolwarm",
    ) -> plotly.graph_objs._figure.Figure:
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
            plotly.graph_objs._figure.Figure:
                The plotly Figure object representing the Sankey diagram.

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
    ) -> plotly.graph_objs._figure.Figure:
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
            plotly.graph_objs._figure.Figure
                The plotly Figure object representing the Sankey diagram.
        """

        fig = complete_sankey(
            self.complete_df,
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
        self.df["fan_tot"] = self.df.apply(lambda x: x["fan_in"] + x["fan_out"], axis=1)
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
