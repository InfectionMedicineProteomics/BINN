import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm


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
):
    """
    Visualizes the most important nodes in a network using a multipartite (layered) layout,
    with a single output node to represent all non-top connections in the final layer.

    - 'Sink nodes' gather all non-top nodes from a given layer and inherit their connections.
      They are drawn at the bottom of each layer with fixed size and color, and are excluded
      from the colormap scaling.

    Parameters:
    - dataframe: Input dataframe containing node information.
      Must have columns: [source_node, target_node, source_layer, target_layer, normalized_importance].
    - top_n: Number of top nodes to visualize per layer. Can be a dictionary {layer: top_n_value}
             or a single integer for all layers.
    - layer_normalization_value: Normalization value for importance in each layer.
    - sink_node_size: Fixed size for sink nodes.
    - sink_node_color: Color for sink nodes.
    - node_cmap: Colormap for non-sink nodes.
    - plot_size: Tuple specifying the plot size (width, height).
    - alpha: Transparency for node colors.
    - edge_color: Color for edges.
    - edge_alpha: Transparency for edges.
    - arrow_size: Arrow size for directed edges.
    - edge_width_scaling: Scaling factor for edge width based on importance.
    - node_size_scaling: Scaling factor for node size based on importance.
    """

    # Ensure layer columns are of consistent type (convert to string)
    dataframe["source_layer"] = dataframe["source_layer"].astype(str)
    dataframe["target_layer"] = dataframe["target_layer"].astype(str)

    # Create unique IDs by merging node and layer
    dataframe["unique_source"] = (
        dataframe["source_node"] + "_" + dataframe["source_layer"]
    )
    dataframe["unique_target"] = (
        dataframe["target_node"] + "_" + dataframe["target_layer"]
    )

    # Aggregate node importance for source nodes
    source_importance_df = (
        dataframe.groupby(["unique_source", "source_layer"])["normalized_importance"]
        .sum()
        .reset_index()
    )

    # Normalize importance values per layer
    source_importance_df["normalized_importance"] = source_importance_df.groupby(
        "source_layer"
    )["normalized_importance"].transform(
        lambda x: x / x.sum() * layer_normalization_value
    )

    # Ensure top_n is a dictionary for layer-specific values
    if isinstance(top_n, int):
        top_n = {
            layer: top_n for layer in source_importance_df["source_layer"].unique()
        }

    # Select top nodes within each layer based on aggregated importance
    top_nodes_by_layer = (
        source_importance_df.groupby("source_layer")
        .apply(lambda x: x.nlargest(top_n.get(x.name, 0), "normalized_importance"))
        .reset_index(drop=True)
    )

    # Get set of top source nodes for quick lookup
    top_source_nodes = set(top_nodes_by_layer["unique_source"])

    # Identify sink nodes, renaming the final sink node to "output_node"
    layers = set(dataframe["source_layer"].unique()).union(
        set(dataframe["target_layer"].unique())
    )
    final_layer = max(int(layer) for layer in layers)
    sink_nodes = {}
    for layer in layers:
        if int(layer) == final_layer:
            sink_nodes[layer] = "output_node"
        else:
            sink_nodes[layer] = f"sink_{layer}"

    # Initialize a directed graph
    G = nx.DiGraph()

    # Keep track of each node's importance.
    # For non-sink nodes, use the top_nodes_by_layer info; initialize sink nodes with 0.
    node_importance = {}
    for _, row in top_nodes_by_layer.iterrows():
        node_importance[row["unique_source"]] = row["normalized_importance"]
    for layer, sink_node in sink_nodes.items():
        node_importance[sink_node] = 0.0

    # ---------------------------------------------------------------------
    # Build edges while mapping non-top nodes to their layer-specific sink node.
    # This way, sink nodes inherit the connections from all underlying non-top nodes.
    # ---------------------------------------------------------------------
    for _, row in dataframe.iterrows():
        source = row["unique_source"]
        target = row["unique_target"]
        s_layer = row["source_layer"]
        t_layer = row["target_layer"]
        importance = row["normalized_importance"]

        # Map to sink node if not top node
        new_source = source if source in top_source_nodes else sink_nodes[s_layer]
        new_target = target if target in top_source_nodes else sink_nodes[t_layer]

        # Create nodes with their corresponding layer if not present
        if not G.has_node(new_source):
            G.add_node(new_source, layer=int(s_layer))
        if not G.has_node(new_target):
            G.add_node(new_target, layer=int(t_layer))

        # Accumulate node importance for both nodes.
        node_importance[new_source] = node_importance.get(new_source, 0) + importance
        node_importance[new_target] = node_importance.get(new_target, 0) + importance

        # Aggregate edge weights if edge exists; otherwise add it.
        if G.has_edge(new_source, new_target):
            G[new_source][new_target]["weight"] += importance
        else:
            G.add_edge(new_source, new_target, weight=importance)

    # ---------------------------------------------------------------------
    # Determine node layers.
    # ---------------------------------------------------------------------
    node_layers = {node: data["layer"] for node, data in G.nodes(data=True)}

    # For laying out nodes, group nodes per layer.
    # Also, adjust ordering so that sink nodes always end up at the bottom.
    layer_nodes = {}
    for node, layer in node_layers.items():
        layer_nodes.setdefault(layer, []).append(node)

    for layer in layer_nodes:
        # Separate non-sink nodes from sink nodes.
        non_sink = [n for n in layer_nodes[layer] if n not in sink_nodes.values()]
        sinks = [n for n in layer_nodes[layer] if n in sink_nodes.values()]
        # Sort non-sink nodes descending by importance.
        non_sink_sorted = sorted(non_sink, key=lambda n: -node_importance.get(n, 0))
        # The final order is non-sink nodes followed by sink nodes.
        layer_nodes[layer] = non_sink_sorted + sinks

    # ---------------------------------------------------------------------
    # Create the layout using NetworkX's multipartite_layout.
    # We'll then adjust vertical positions for each layer so that they are centered.
    # ---------------------------------------------------------------------
    pos = nx.multipartite_layout(G, subset_key="layer")

    for layer, nodes_in_layer in layer_nodes.items():
        count = len(nodes_in_layer)
        # Compute an offset so that positions will be centered:
        # For example, if count=3, we want y-coords: 1, 0, -1.
        offset = (count - 1) / 2
        for idx, node in enumerate(nodes_in_layer):
            # Here, a higher index means a lower position.
            pos[node] = (layer, offset - idx)

    # ---------------------------------------------------------------------
    # Create normalization for colormap using only non-sink nodes per layer.
    # ---------------------------------------------------------------------
    layer_norms = {}
    for layer, nodes_in_layer in layer_nodes.items():
        # Only consider non-sink nodes for colormap normalization.
        non_sink = [n for n in nodes_in_layer if n not in sink_nodes.values()]
        if non_sink:
            values = [node_importance[n] for n in non_sink]
            layer_norms[layer] = mcolors.Normalize(vmin=min(values), vmax=max(values))
        else:
            layer_norms[layer] = mcolors.Normalize(vmin=0, vmax=1)

    # ---------------------------------------------------------------------
    # Plot the graph.
    # ---------------------------------------------------------------------
    plt.figure(figsize=plot_size)
    cmap = plt.get_cmap(node_cmap)

    for layer, nodes_in_layer in layer_nodes.items():
        norm = layer_norms[layer]
        sizes = []
        colors = []
        for n in nodes_in_layer:
            if n in sink_nodes.values():  # Fixed size and color for sink nodes.
                sizes.append(sink_node_size)
                colors.append(sink_node_color)
            else:
                sizes.append(node_importance.get(n, 0) * node_size_scaling)
                colors.append(cmap(norm(node_importance.get(n, 0))))
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_in_layer,
            node_size=sizes,
            node_color=colors,
            alpha=alpha,
        )

    # Draw edges with widths based on importance.
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=arrow_size,
        edge_color=edge_color,
        alpha=edge_alpha,
        width=edge_width,
    )

    # Draw node labels.
    nx.draw_networkx_labels(G, pos)

    # Add a colorbar for non-sink nodes (using the last computed cmap).
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="Normalized node importance")

    plt.axis("off")
    plt.tight_layout()
    return plt
