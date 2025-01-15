from typing import Union
from binn.analysis.plot import subgraph_sankey, complete_sankey
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
        norm_method: str = "subgraph",
        val_col: str = "value",
    ):
        self.root_node = 0
        self.importance_df = importance_df
        self.val_col = val_col
        self.importance_graph = self.create_graph()
        self.importance_graph_reverse = self.importance_graph.reverse()
        self.norm_method = norm_method
        if norm_method:
            self.importance_df = self.add_normalization(method=norm_method)

    

    
    

    def plot_subgraph_sankey(
        self,
        query_node: str,
        upstream: bool = False,
        savename: str = "sankey.png",
        val_col: str = "value",
        cmap: str = "coolwarm",
        width: int = 1200,
        scale: float = 2.5,
        height: int = 500,
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
        if upstream:
            final_node_id = self.get_node(query_node)
            subgraph = self.get_upstream_subgraph(final_node_id, depth_limit=None)
            source_or_target = "target"
        else:
            if query_node == "root":
                return ValueError("You cannot look downstream from root")
            final_node_id = self.get_node("root")
            query_node_id = self.get_node(query_node)
            subgraph = self.get_downstream_subgraph(self.importance_graph, query_node_id, depth_limit=None)
            source_or_target = "source"
            
        nodes_in_subgraph = [n for n in subgraph.nodes]

        df = self.importance_df[
            self.importance_df[source_or_target].isin(nodes_in_subgraph)
        ].copy()

        if df.empty:
            return ValueError("There are no nodes in the specified subgraph")

        fig = subgraph_sankey(
            df, final_node=final_node_id, val_col=val_col, cmap_name=cmap
        )

        fig.write_image(f"{savename}", width=width, scale=scale, height=height)
        return fig

    def plot_complete_sankey(
        self,
        multiclass: bool = False,
        show_top_n: int = 10,
        node_cmap: str = "Reds",
        edge_cmap: Union[str, list] = "Reds",
        savename="sankey.png",
        width: int = 1900,
        scale: float = 2,
        height: int = 800,
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

        fig.write_image(f"{savename}", width=width, scale=scale, height=height)
        return fig
