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


def get_mapping_to_all_layers(path_df, translation_df):
    graph = nx.from_pandas_edgelist(
        path_df, source="child", target="parent", create_using=nx.DiGraph()
    )

    components = {"input": [], "connections": []}
    for translation in translation_df["input"]:
        ids = translation_df[translation_df["input"]
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


class Network:
    def __init__(
        self,
        input_data: pd.DataFrame,
        pathways: pd.DataFrame,
        mapping: Union[pd.DataFrame, None] = None,
        input_data_column="Protein",
        verbose: bool = False,
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
                input_data, self.mapping, input_data_column, verbose
            )

            self.pathways = subset_pathways_on_idx(
                pathways, self.mapping, verbose)

        else:

            self.pathways = pathways

        self.mapping = get_mapping_to_all_layers(self.pathways, self.mapping)

        self.input_data = input_data

        self.inputs = self.mapping["input"].unique()

        self.netx = self.build_network()

    def get_terminals(self):
        return [n for n, d in self.netx.out_degree() if d == 0]

    def get_roots(self):
        return get_nodes_at_level(self.netx, distance=1)

    # get a DiGraph representation of the Reactome hierarchy
    def build_network(self):

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

        return nx.bfs_tree(
            self.netx, "root"
        )

    def get_completed_network(self, n_levels):
        return complete_network(self.netx, n_levels=n_levels)

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        return complete_network(G, n_levels=n_levels)

    def get_layers(self, n_levels, direction="root_to_leaf"):
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

    def get_connectivity_matrices(self, n_levels, direction="root_to_leaf"):

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
    input_df, translation_df, input_data_column, verbose=False
):
    keys_in_data = input_df[input_data_column].unique()

    translation_df = translation_df[translation_df["input"].isin(keys_in_data)]

    if verbose:
        print(f"Number of ids before subsetting: {len(translation_df.index)}")
        print(
            f"Unique ids in df: {len(list(translation_df['input'].unique()))}"
        )
    return translation_df


def subset_pathways_on_idx(path_df, translation_df, verbose=False):
    """
    Recursive method to add parents and children to pathway_df based on filtered translation_df.
    """

    def add_pathways(counter, idx_list, parent):
        counter += 1
        if verbose:
            print(f"Function called {counter} times.")
            print(f"Values in idx_list: {len(idx_list)}")
        if len(parent) == 0:
            print("Base case reached")
            return idx_list
        else:
            idx_list = idx_list + parent
            subsetted_pathway = path_df[path_df["child"].isin(parent)]
            new_parent = list(subsetted_pathway["parent"].unique())
            return add_pathways(counter, idx_list, new_parent)

    counter = 0
    original_parent = list(translation_df["translation"].unique())
    idx_list = []
    idx_list = add_pathways(counter, idx_list, original_parent)
    path_df = path_df[path_df["child"].isin(idx_list)]
    print("Final number of unique connections in pathway: ", len(path_df.index))
    return path_df


class ImportanceNetwork:
    def __init__(self, df: pd.DataFrame, val_col: str = "value"):
        self.df = df
        self.val_col = val_col
        self.G = self.create_graph()
        self.G_reverse = self.G.reverse()

    def create_graph(self):
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

    def get_downstream_subgraph(self, query_node, depth_limit=None):
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

    def get_upstream_subgraph(self, query_node, depth_limit=None):
        SG = self.get_downstream_subgraph(query_node, depth_limit=depth_limit)
        return SG

    def get_complete_subgraph(self, query_node, depth_limit=None):
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

    def get_nr_nodes_in_upstream_SG(self, query_node):
        SG = self.get_upstream_subgraph(query_node, depth_limit=None)
        return SG.number_of_nodes()

    def get_nr_nodes_in_downstream_SG(self, query_node):
        SG = self.get_downstream_subgraph(query_node, depth_limit=None)
        return SG.number_of_nodes()

    def get_fan_in(self, query_node):
        return len([n for n in self.G.in_edges(query_node)])

    def get_fan_out(self, query_node):
        return len([n for n in self.G.out_edges(query_node)])

    def add_normalization(self):
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
        query_node,
        upstream=False,
        savename="sankey.png",
        val_col="value",
        cmap_name="coolwarm",
    ):
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
