from typing import Union

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.graph_objects as go


def plot_sankey(
    df: pd.DataFrame,
    multiclass: bool = False,
    show_top_n: int = 10,
    val_col: str = "importance",
    node_cmap: str = "Reds",
    edge_cmap: Union[str, list] = "coolwarm",
):
    """
    Example df columns (one row per edge):
    source_layer, target_layer, source_node, target_node, class_idx, importance, source_id, target_id, normalized_importance
    
    We group edges per (source_id, target_id) if not multiclass. Then we rename all "low-importance" nodes
    into "other" nodes of the form: nOther_l<layer>. Finally, we build a Sankey chart via Plotly.
    """

    # Use a string prefix for 'other' nodes:
    # We'll append "_l<layer>" where <layer> is either source or target
    # so that each layer’s "other" node has a distinct label
    other_id = "nOther"

    # --- Preprocessing
    # if not multiclass, sum up importance by source->target
    if not multiclass:
        df = df.groupby(
            by=["source_id", "target_id", "source_node", "target_node"], as_index=False
        ).agg(
            {
                val_col: "sum",
                "source_layer": "mean",
                "target_layer": "mean",
                "class_idx": "mean",
            }
        )

    df["source_layer"] = df["source_layer"].astype(int)
    df["target_layer"] = df["target_layer"].astype(int)
    df = _remove_loops(df)
    df["value"] = df[val_col]

    # figure out how many layers there are
    n_layers = df["target_layer"].max()

    # name_map is used later to create the display names
    name_map = _create_name_map(df, n_layers, other_id)

    # --- Build up a top-n list for each layer. Nodes outside top_n become "nOther_l<layer>".
    top_n = {}
    for layer in range(n_layers):
        top_n[layer] = (
            df.loc[df["source_layer"] == layer]
            .groupby("source_id", as_index=True)[["value"]]
            .mean(numeric_only=True)
            .sort_values("value", ascending=False)
            .iloc[:show_top_n]
            .index.tolist()
        )

    def set_to_other(row, top_n: dict, source_or_target: str):
        """If the node is not in top_n, rename it as nOther_l<layer>."""
        node = row[source_or_target]
        # We define the layer offset so that for target we do an additional +1
        # so that "layer" matches the node's "layer" concept more closely.
        layer = row["source_layer"] + 1
        if source_or_target == "target_id":
            layer += 1

        # If in top_n for any layer, keep it.
        for layer_values in top_n.values():
            if node in layer_values:
                return node

        # Otherwise, rename to e.g. "nOther_l2"
        return f"{other_id}_l{layer}"

    # Convert nodes outside top_n to "nOther_l<layer>"
    df["source_w_other"] = df.apply(
        lambda x: set_to_other(x, top_n, "source_id"), axis=1
    )
    df["target_w_other"] = df.apply(
        lambda x: set_to_other(x, top_n, "target_id"), axis=1
    )

    # --- Normalize, giving "other" nodes a scaled-down contribution if desired
    def normalize_layer_values(df: pd.DataFrame):
        """
        For "other" nodes, we scale the value differently (e.g. by 0.1)
        so that they don't dominate the visualization.
        """
        new_df = pd.DataFrame()
        # Mark which rows are "other" by checking if they start with "nOther"
        df["Other"] = df["source_w_other"].apply(lambda x: x.startswith(other_id))

        # separate out the "other" rows vs "non-other" rows
        other_df = df[df["Other"]].copy()
        df = df[~df["Other"]].copy()

        # normalize within each (source_layer)
        for layer in df["source_layer"].unique():
            layer_df = df[df["source_layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized_value"] = (
                layer_df["value"] / layer_total if layer_total != 0 else 0
            )
            new_df = pd.concat([new_df, layer_df], ignore_index=True)

        # scale "other" rows differently
        for layer in other_df["source_layer"].unique():
            layer_df = other_df[other_df["source_layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized_value"] = (
                0.1 * layer_df["value"] / layer_total if layer_total != 0 else 0
            )
            new_df = pd.concat([new_df, layer_df], ignore_index=True)

        return new_df

    df = normalize_layer_values(df)

    # sum again by (source_w_other, target_w_other, class_idx)
    df = df.groupby(
        by=["source_w_other", "target_w_other", "class_idx"], sort=False, as_index=False
    ).agg(
        {
            "normalized_value": "sum",
            "value": "sum",
            "source_layer": "mean",
            "target_layer": "mean",
        }
    )

    # prepare the feature encoding
    unique_features = (
        df["source_w_other"].unique().tolist()
        + df["target_w_other"].unique().tolist()
    )
    # drop duplicates
    unique_features = list(set(unique_features))

    code_map, feature_labels = _encode_features(unique_features)
    
    # --- Next, build link arrays
    def get_connections(sources: list, df: pd.DataFrame):
        """
        Build up source/target/value/link_color arrays in the order of the row data.
        """
        # Filter df by only the specified sources
        conn = df[df["source_w_other"].isin(sources)].copy()

        source_code = [_get_code(s, code_map) for s in conn["source_w_other"]]
        target_code = [_get_code(t, code_map) for t in conn["target_w_other"]]
        values = conn["normalized_value"].tolist()

        if not multiclass:
            # single color scale
            temp_df, _ = get_node_colors(feature_labels, df, curr_cmap=edge_cmap)
            # quick hack: just pick one color per row.
            # (Or you could do something fancier with the numeric magnitude, etc.)
            link_colors = (
                temp_df["node_color"]
                .apply(lambda x: x.split(", 0.75)")[0] + ", 0.75)")
                .tolist()
            )
        else:
            # if multiclass, maybe pick from list of colors
            link_colors = conn.apply(
                lambda x: (
                    "rgba(236,236,236, 0.75)"
                    if x["source_w_other"].startswith(other_id)
                    else edge_cmap[int(x["class_idx"]) % len(edge_cmap)]
                ),
                axis=1,
            ).tolist()
        return source_code, target_code, values, link_colors

    def get_node_colors(sources: list, df: pd.DataFrame, curr_cmap=node_cmap):
        """
        Color each node according to a colormap based on the node's average normalized_value.
        'Other' nodes become light grey.
        """
        # Build a per-layer color map
        cmaps = {}
        for layer in df["source_layer"].unique():
            c_df = df[df["source_layer"] == layer].copy()
            # only real nodes for min/max range
            c_df = c_df[~c_df["source_w_other"].str.startswith(other_id)]
            if not c_df.empty:
                vmin = c_df.groupby("source_w_other")["normalized_value"].mean().min()
                vmax = c_df.groupby("source_w_other")["normalized_value"].mean().max()
            else:
                # fallback
                vmin, vmax = 0, 1
            cmap = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=vmin * 0.8, vmax=vmax),
                cmap=curr_cmap,
            )
            cmaps[layer] = cmap

        colors = []
        new_df = pd.DataFrame()
        for s in sources:
            source_df = df[df["source_w_other"] == s].copy()
            if s.startswith(other_id):
                # color for "other" nodes => light grey
                c = "rgba(236,236,236, 0.75)"
                colors.append(c)
                source_df["node_color"] = c
            else:
                # normal node
                # pick its average normalized_value
                intensity = source_df["normalized_value"].mean()
                layer_val = int(source_df["source_layer"].mean())
                cmap = cmaps.get(layer_val)
                if cmap is None:
                    # fallback
                    c = "rgba(200,200,200, 0.75)"
                else:
                    r, g, b, a = cmap.to_rgba(intensity, alpha=0.75)
                    c = f"rgba({r*255:.0f}, {g*255:.0f}, {b*255:.0f}, {a:.2f})"
                colors.append(c)
                source_df["node_color"] = c
            new_df = pd.concat([new_df, source_df], ignore_index=True)

        return new_df, colors

    def get_node_positions(feature_labels: list, df: pd.DataFrame):
        """
        For each layer, compute x,y positions so that 'nOther' lumps get placed at the same y, 
        and real nodes get spread out from top to bottom.
        """
        # group by the node name
        grouped_df = df.groupby("source_w_other", as_index=False).agg(
            {"source_layer": "min", "value": "mean"}
        )

        # We’ll store final x,y in a DataFrame
        final_df = pd.DataFrame()
        layers = list(range(n_layers))

        for layer in layers:
            # get only nodes from that layer
            layer_df = grouped_df[grouped_df["source_layer"] == layer].copy()
            # separate "other" from "real" nodes
            other_df = layer_df[layer_df["source_w_other"].str.startswith(other_id)].copy()
            layer_df = layer_df[~layer_df["source_w_other"].str.startswith(other_id)].copy()

            # sort real nodes by their "value" so we can place them top->bottom
            layer_df = layer_df.sort_values("value", ascending=True)
            layer_df["rank"] = range(len(layer_df))
            # normalize the sum so bigger-value nodes appear a bit higher or lower
            if layer_df["value"].sum() != 0:
                layer_df["value"] = layer_df["value"] / layer_df["value"].sum()

            # place them from ~0.8 down to ~0
            if len(layer_df) > 1:
                max_rank = layer_df["rank"].max()
                layer_df["y"] = 0.8 * (max_rank - layer_df["rank"]) / (max_rank + 0.001)
            else:
                # if only 1 node, place in the middle
                layer_df["y"] = 0.4

            layer_df["x"] = (0.01 + layer) / (len(layers) + 1)

            # For the "other" nodes, we collapse them into a single row
            # with some default y=0.9, rank=999, etc.
            if not other_df.empty:
                other_value_sum = other_df["value"].sum()
                row = pd.DataFrame(
                    [
                        [
                            f"{other_id}_l{layer}",
                            layer,
                            other_value_sum,
                            999,
                            0.9,
                            (0.01 + layer) / (len(layers) + 1),
                        ]
                    ],
                    columns=["source_w_other", "source_layer", "value", "rank", "y", "x"],
                )
                layer_df = pd.concat([layer_df, row], ignore_index=True)

            final_df = pd.concat([final_df, layer_df], ignore_index=True)

        # Now build x,y arrays in the same order as feature_labels
        x, y = [], []
        for label in feature_labels:
            node_row = final_df[final_df["source_w_other"] == label]
            if node_row.empty:
                # default coords if not found
                x.append([0.5])
                y.append([0.5])
            else:
                x.append(node_row["x"].values)
                y.append(node_row["y"].values)
        return x, y

    # --- Build up the data for Plotly sankey
    # 1) get node colors
    df_with_colors, node_colors = get_node_colors(feature_labels, df)

    # 2) get link arrays
    sources = df["source_w_other"].unique().tolist()
    encoded_source, encoded_target, value, link_colors = get_connections(sources, df)

    # 3) positions
    x, y = get_node_positions(feature_labels, df)

    # 4) rename labels using name_map if possible
    new_labels = []
    for label in feature_labels:
        if label in name_map:
            new_labels.append(name_map[label])
        else:
            new_labels.append(label)

    # Build the Sankey figure
    nodes = dict(
        pad=15,
        thickness=15,
        line=dict(color="white", width=0),
        label=new_labels,
        color=node_colors,
        x=x,
        y=y,
    )
    links = dict(
        source=encoded_source,
        target=encoded_target,
        value=value,
        color=link_colors,
    )

    fig = go.Figure(
        data=[
            go.Sankey(
                textfont=dict(size=15, family="Arial"),
                orientation="h",
                arrangement="snap",
                node=nodes,
                link=links,
            )
        ],
    )
    return fig


def _encode_features(features: list):
    """Assign each unique feature string an integer code."""
    feature_map = pd.DataFrame({
        "feature": features,
        "code": list(range(len(features)))
    })
    return feature_map, features


def _get_code(feature: str, feature_map: pd.DataFrame):
    """Look up the integer code for the given feature string."""
    return feature_map.loc[feature_map["feature"] == feature, "code"].values[0]


def _remove_loops(df: pd.DataFrame):
    """
    Drop any edges where source_node == target_node (self loops).
    """
    df["loop"] = df.apply(lambda x: x["source_node"] == x["target_node"], axis=1)
    df = df[df["loop"] == False].copy()
    df.drop(columns=["loop"], inplace=True)
    return df


def _create_name_map(df: pd.DataFrame, n_layers: int, other_id: str):
    """
    Build a map from raw IDs (like "nA0M8Q6_l0") to a more readable name (like "A0M8Q6"),
    plus special mapping for 'other' nodes, e.g. "nOther_l2" -> "Other connections 2".
    """
    # Basic map: source_id -> source_node
    name_map = dict(
        zip(df["source_id"].tolist(), df["source_node"].tolist())
    )

    # Also define the "nOther_lX" -> "Other connections X" mapping
    for i in range(1, n_layers + 3):
        key = f"{other_id}_l{i}"
        name_map[key] = f"Other connections {i}"

    return name_map
