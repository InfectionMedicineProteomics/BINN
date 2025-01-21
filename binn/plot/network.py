import textwrap
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd

# Import our utility helpers
from binn.plot.util import (
    load_default_mapping,
    build_mapping_dict,
    rename_node_by_layer
)

def visualize_binn(
    dataframe,
    top_n=5,
    layer_normalization_value=1,
    sink_node_size=500,
    sink_node_color="lightgray",
    node_cmap="viridis",
    plot_size=(12, 8),
    alpha=0.9,
    edge_color="gray",
    edge_alpha=0.7,
    arrow_size=10,
    edge_width=1,
    node_size_scaling=10,
    input_entity_mapping=None,  # can be a str, a DataFrame, or None
    pathways_mapping=None,      # can be a str, a DataFrame, or None
    input_entity_key_col="input_id",      # which column to use as key if user-provided
    input_entity_val_col="input_name",       # which column to use as value if user-provided
    pathways_key_col="pathway_id",    # which column to use as key if user-provided
    pathways_val_col="pathway_name",           # which column to use as value if user-provided
):
    """
    Visualizes the most important nodes in a network using a multipartite (layered) layout,
    with a single output node to represent all non-top connections in the final layer.

    'Sink nodes' gather all non-top nodes from a given layer and inherit their connections.
    They are drawn at the bottom of each layer with fixed size and color, and are excluded
    from the colormap scaling.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        Must have columns: 
            [source_node, target_node, source_layer, target_layer, normalized_importance].
    top_n : int or dict
        Number of top nodes to visualize per layer. 
        If an int, applies to all layers. If dict, keys are layers and values are ints.
    layer_normalization_value : float
        Normalization value for importance in each layer.
    sink_node_size : float
        Fixed size for sink nodes.
    sink_node_color : str
        Color for sink nodes.
    node_cmap : str
        Colormap name for non-sink nodes.
    plot_size : tuple
        (width, height) for the figure.
    alpha : float
        Transparency for node colors.
    edge_color : str
        Color for edges.
    edge_alpha : float
        Transparency for edges.
    arrow_size : int
        Size of arrows in the drawn edges.
    edge_width : float
        Width of edges.
    node_size_scaling : float
        Scaling factor for node size based on importance.

    input_entity_mapping : {None, str, pd.DataFrame}
        If None, no mapping is done for layer 1.
        If str (e.g. "reactome"), we attempt to load a default DataFrame 
        from the package (not typical for input entities, but flexible).
        If pd.DataFrame, we build a dictionary for renaming.

    pathways_mapping : {None, str, pd.DataFrame}
        If None, no mapping is done for layers != 1.
        If str (e.g. "reactome"), we attempt to load a default DataFrame 
        from the package.
        If pd.DataFrame, we build a dictionary for renaming.

    input_entity_key_col : str
        Column name used as the key in input_entity_mapping if it's a DataFrame.
    input_entity_val_col : str
        Column name used as the value in input_entity_mapping.

    pathways_key_col : str
        Column name used as the key in pathways_mapping if it's a DataFrame.
    pathways_val_col : str
        Column name used as the value in pathways_mapping.
    """

    # ---------------------------------------------------------------------
    # 1) Build or load dictionaries for input entities and pathways
    # ---------------------------------------------------------------------
    # input entities (usually for layer 1)
    if isinstance(input_entity_mapping, str):
        # load default df, then build dict
        input_entity_df = load_default_mapping(input_entity_mapping)
        input_entity_dict = build_mapping_dict(
            input_entity_df, key_col=input_entity_key_col, val_col=input_entity_val_col
        )
    elif isinstance(input_entity_mapping, pd.DataFrame):
        # user-provided df => build dict
        input_entity_dict = build_mapping_dict(
            input_entity_mapping, key_col=input_entity_key_col, val_col=input_entity_val_col
        )
    else:
        # None or anything else => no mapping
        input_entity_dict = None

    # pathways (usually for layers != 1)
    if isinstance(pathways_mapping, str):
        # load default df, then build dict
        pathways_df = load_default_mapping(pathways_mapping)
        pathways_dict = build_mapping_dict(
            pathways_df, key_col=pathways_key_col, val_col=pathways_val_col
        )
    elif isinstance(pathways_mapping, pd.DataFrame):
        # user-provided df => build dict
        pathways_dict = build_mapping_dict(
            pathways_mapping, key_col=pathways_key_col, val_col=pathways_val_col
        )
    else:
        pathways_dict = None

    # ---------------------------------------------------------------------
    # 2) Preprocess the main dataframe: ensure layers are strings, build (node, layer) tuples
    # ---------------------------------------------------------------------
    dataframe["source_layer"] = dataframe["source_layer"].astype(str)
    dataframe["target_layer"] = dataframe["target_layer"].astype(str)

    dataframe["source_tuple"] = list(zip(dataframe["source_node"], dataframe["source_layer"]))
    dataframe["target_tuple"] = list(zip(dataframe["target_node"], dataframe["target_layer"]))

    # ---------------------------------------------------------------------
    # 3) Aggregate + normalize node importance
    # ---------------------------------------------------------------------
    if "normalized_importance" in dataframe.columns:
        value_column = "normalized_importance"
    else:
        value_column = "importance"

    source_importance_df = (
        dataframe
        .groupby(["source_tuple", "source_layer"])[value_column]
        .sum()
        .reset_index()
    )

    # Normalize importance by layer
    source_importance_df[value_column] = source_importance_df.groupby("source_layer")[
        value_column
    ].transform(lambda x: x / x.sum() * layer_normalization_value)

    # Handle top_n as dict
    if isinstance(top_n, int):
        layers_unique = source_importance_df["source_layer"].unique()
        top_n = {layer: top_n for layer in layers_unique}

    def pick_top(group):
        n = top_n.get(group.name, 0)
        return group.nlargest(n, value_column)

    top_nodes_by_layer = (
        source_importance_df
        .groupby("source_layer")
        .apply(pick_top)
        .reset_index(drop=True)
    )
    top_source_nodes = set(top_nodes_by_layer["source_tuple"])

    # ---------------------------------------------------------------------
    # 4) Identify sink nodes
    # ---------------------------------------------------------------------
    all_layers = set(dataframe["source_layer"].unique()).union(
        set(dataframe["target_layer"].unique())
    )
    final_layer = max(int(layer) for layer in all_layers)

    # We'll call them ("sink", layer) except for final layer => ("output_node", final_layer)
    sink_nodes = {}
    for layer_str in all_layers:
        if int(layer_str) == final_layer:
            sink_nodes[layer_str] = ("output_node", layer_str)
        else:
            sink_nodes[layer_str] = ("sink", layer_str)

    # ---------------------------------------------------------------------
    # 5) Build the directed graph
    # ---------------------------------------------------------------------
    G = nx.DiGraph()
    node_importance = {}

    # initialize top-node importance
    for _, row in top_nodes_by_layer.iterrows():
        node_tuple = row["source_tuple"]
        node_importance[node_tuple] = row[value_column]

    # initialize sink-node importance
    for layer_str, sink_tuple in sink_nodes.items():
        node_importance[sink_tuple] = 0.0

    # For each row in the dataframe, redirect non-top nodes to sink nodes
    for _, row in dataframe.iterrows():
        s_tuple = row["source_tuple"]
        t_tuple = row["target_tuple"]
        s_layer = row["source_layer"]
        t_layer = row["target_layer"]
        importance = row[value_column]

        new_source = s_tuple if s_tuple in top_source_nodes else sink_nodes[s_layer]
        new_target = t_tuple if t_tuple in top_source_nodes else sink_nodes[t_layer]

        if not G.has_node(new_source):
            G.add_node(
                new_source,
                layer=int(s_layer),
                name=rename_node_by_layer(
                    new_source, 
                    input_entity_dict=input_entity_dict, 
                    pathways_dict=pathways_dict
                ),
            )
        if not G.has_node(new_target):
            G.add_node(
                new_target,
                layer=int(t_layer),
                name=rename_node_by_layer(
                    new_target, 
                    input_entity_dict=input_entity_dict, 
                    pathways_dict=pathways_dict
                ),
            )

        node_importance[new_source] = node_importance.get(new_source, 0) + importance
        node_importance[new_target] = node_importance.get(new_target, 0) + importance

        if G.has_edge(new_source, new_target):
            G[new_source][new_target]["weight"] += importance
        else:
            G.add_edge(new_source, new_target, weight=importance)

    # ---------------------------------------------------------------------
    # 6) Figure out node order per layer and generate layout
    # ---------------------------------------------------------------------
    node_layers = {node: data["layer"] for node, data in G.nodes(data=True)}
    layer_nodes = {}
    for node, layer in node_layers.items():
        layer_nodes.setdefault(layer, []).append(node)

    # Sort non-sink nodes within each layer by descending importance
    for layer in layer_nodes:
        sinks_here = [n for n in layer_nodes[layer] if n in sink_nodes.values()]
        non_sink = [n for n in layer_nodes[layer] if n not in sinks_here]
        non_sink_sorted = sorted(non_sink, key=lambda n: -node_importance.get(n, 0))
        layer_nodes[layer] = non_sink_sorted + sinks_here

    pos = nx.multipartite_layout(G, subset_key="layer")

    # Re-center vertically
    for layer, nodes_in_layer in layer_nodes.items():
        count = len(nodes_in_layer)
        offset = (count - 1) / 2.0
        for idx, node in enumerate(nodes_in_layer):
            pos[node] = (layer, offset - idx)

    # ---------------------------------------------------------------------
    # 7) Colormap normalization (exclude sink nodes)
    # ---------------------------------------------------------------------
    cmap = plt.get_cmap(node_cmap)
    layer_norms = {}
    for layer, nodes_in_layer in layer_nodes.items():
        non_sink = [n for n in nodes_in_layer if n not in sink_nodes.values()]
        if non_sink:
            vals = [node_importance[n] for n in non_sink]
            layer_norms[layer] = mcolors.Normalize(vmin=min(vals), vmax=max(vals))
        else:
            layer_norms[layer] = mcolors.Normalize(vmin=0, vmax=1)

    # ---------------------------------------------------------------------
    # 8) Plot
    # ---------------------------------------------------------------------
    plt.figure(figsize=plot_size)

    # Draw nodes
    for layer, nodes_in_layer in layer_nodes.items():
        norm = layer_norms[layer]
        sizes = []
        colors = []
        for n in nodes_in_layer:
            if n in sink_nodes.values():
                sizes.append(sink_node_size)
                colors.append(sink_node_color)
            else:
                val = node_importance.get(n, 0)
                sizes.append(val * node_size_scaling)
                colors.append(cmap(norm(val)))
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_in_layer,
            node_size=sizes,
            node_color=colors,
            alpha=alpha,
        )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=arrow_size,
        edge_color=edge_color,
        alpha=edge_alpha,
        width=edge_width,
    )

    # Draw labels (the "name" attribute we set above)
    max_label_width = 20
    labels = {
        node: "\n".join(textwrap.wrap(data["name"], width=max_label_width))
        for node, data in G.nodes(data=True)
    }
    nx.draw_networkx_labels(G, pos, labels=labels)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="Normalized node importance")

    plt.axis("off")
    plt.tight_layout()
    return plt
