from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.graph_objects as go


def subgraph_sankey(
    df: pd.DataFrame,
    final_node: int = 0,
    val_col="value",
    cmap_name="coolwarm",
):
    """
    Create a Sankey diagram using Plotly and Seaborn.
    Parameters:
        df : pd.DataFrame
            The input DataFrame containing the data to be plotted.
        final_node : str, optional
            The final node in the Sankey diagram (default is "root").
        val_col : str, optional
            The column in the input DataFrame containing the values to be plotted (default is "value").
        cmap_name : str, optional
            The name of the color map to be used (default is "coolwarm").
    Returns:
        fig : go.Figure
            The Sankey diagram figure created using Plotly.
    """
    df["source layer"] = df["source layer"].astype(int)
    df["target layer"] = df["target layer"].astype(int)
    unique_features = df["source"].unique().tolist()
    unique_features += df["target"].unique().tolist()
    code_map, feature_labels = _encode_features(list(set(unique_features)))
    sources = df["source"].unique().tolist()
    name_map = _create_subgraph_name_map(df)

    def normalize_layer_values(df: pd.DataFrame):
        new_df = pd.DataFrame()
        total_value_sum = df[val_col].sum()
        for layer in df["source layer"].unique():
            layer_df = df[df["source layer"] == layer].copy()
            layer_total = layer_df[val_col].sum()
            layer_df.loc[:, "normalized value"] = (
                total_value_sum * layer_df[val_col] / layer_total
            )
            new_df = pd.concat([new_df, layer_df])
        return new_df

    df = _remove_loops(df)
    df = normalize_layer_values(df)

    def get_connections(sources: list, source_target_df: pd.DataFrame):
        conn = source_target_df[source_target_df["source"].isin(sources)]
        source_code = [_get_code(s, code_map) for s in conn["source"]]
        target_code = [_get_code(s, code_map) for s in conn["target"]]
        values = [v for v in conn["normalized value"]]
        link_colors = conn["node_color"].values.tolist()
        return source_code, target_code, values, link_colors

    def get_node_colors(sources, df: pd.DataFrame):
        cmap = plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=cmap_name
        )
        colors = []
        new_df = df
        node_dict = {}
        weight_dict = {}
        for layer in df["source layer"].unique():
            w = df.loc[df["source layer"] == layer, "normalized value"].values
            n = df.loc[df["source layer"] == layer, "source"].values

            if len(w) < 2:
                w = [1]
            else:
                xmin = min(w)
                xmax = max(w)
                if xmax == xmin:
                    w = [1] * len(n)
                else:
                    w = np.array(w)
                    X_std = (w - xmin) / (xmax - xmin)
                    X_scaled = X_std
                    w = X_scaled.tolist()

            pairs = [(f, v) for f, v in zip(n, w)]
            for pair in pairs:
                n, w = pair
                r, g, b, a = cmap.to_rgba(w, alpha=0.5)
                weight_dict[n] = w
                node_dict[n] = f"rgba({r * 255}, {g * 255}, {b * 255}, {a})"
        node_dict[final_node] = "rgba(0,0,0,1)"
        weight_dict[final_node] = 1
        colors = [node_dict[n] for n in sources]
        new_df = new_df.assign(
            node_color=[node_dict[n] for n in new_df["source"]],
            node_weight=[weight_dict[n] for n in new_df["source"]],
        )
        return new_df, colors

    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(sources, df)

    new_labels = []
    for label in feature_labels:
        if label in name_map.keys():
            new_labels.append(name_map[label])
        else:
            new_labels.append(label)

    nodes = dict(
        pad=20,
        thickness=20,
        line=dict(color="white", width=2),
        label=new_labels,
        color=node_colors,
    )

    links = dict(
        source=encoded_source, target=encoded_target, value=value, color=link_colors
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


def complete_sankey(
    df: pd.DataFrame,
    multiclass: bool = False,
    show_top_n: int = 10,
    val_col: str = "value",
    node_cmap: str = "Reds",
    edge_cmap: Union[str, list] = "coolwarm",
    root_id: int = 0,
    other_id: int = -1,
):
    if not multiclass:
        df = df.groupby(
            by=["source", "target", "source name", "target name"], as_index=False
        ).agg(
            {
                "value": "sum",
                "source layer": "mean",
                "target layer": "mean",
                "type": "mean",
            }
        )

    df["source layer"] = df["source layer"].astype(int)
    df["target layer"] = df["target layer"].astype(int)
    df = _remove_loops(df)
    n_layers = max(df["target layer"].values)
    df["value"] = df[val_col]
    name_map = _create_name_map(df, n_layers, root_id, other_id)

    top_n = {}

    for layer in range(n_layers):
        top_n[layer] = (
            df.loc[df["source layer"] == layer]
            .groupby("source")
            .mean(numeric_only=True)
            .sort_values("value", ascending=False)
            .iloc[:show_top_n]
            .index.tolist()
        )

    def set_to_other(row, top_n: dict, source_or_target: str):
        node = row[source_or_target]
        layer = row["source layer"] + 1
        if source_or_target == "target":
            layer = layer + 1
        for top_layer_values in top_n.values():
            if node in top_layer_values:
                return node
        if node == root_id:
            return node
        return other_id * layer  # All other nodes are multiples of other_id

    def normalize_layer_values(df: pd.DataFrame):
        new_df = pd.DataFrame()
        df["Other"] = (
            df["source_w_other"]
            .apply(lambda x: True if x <= other_id else False)
            .copy()
        )
        other_df = df[df["Other"] == True]
        df = df[df["Other"] == False]
        for layer in df["source layer"].unique():
            layer_df = df[df["source layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        for layer in other_df["source layer"].unique():
            layer_df = other_df[other_df["source layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = 0.1 * layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        return new_df

    def get_connections(sources: list, df: pd.DataFrame):
        conn = df[df["source_w_other"].isin(sources)].copy()
        source_code = [_get_code(s, code_map) for s in conn["source_w_other"]]
        target_code = [_get_code(s, code_map) for s in conn["target_w_other"]]
        values = [v for v in conn["normalized value"]] * 10
        if multiclass == False:
            temp_df, _ = get_node_colors(feature_labels, df, curr_cmap=edge_cmap)
            link_colors = (
                temp_df["node_color"]
                .apply(lambda x: x.split(", 0.75)")[0] + ", 0.75)")
                .values.tolist()
            )
        else:
            link_colors = conn.apply(
                lambda x: "rgba(236,236,236, 0.75)"
                if x["source_w_other"] <= other_id
                else edge_cmap[x["type"]],
                axis=1,
            ).values.tolist()
        return source_code, target_code, values, link_colors

    def get_node_colors(sources: list, df: pd.DataFrame, curr_cmap=node_cmap):
        cmaps = {}
        for layer in df["source layer"].unique():
            c_df = df[df["source layer"] == layer].copy()
            c_df = c_df[~c_df["source_w_other"] <= other_id]
            cmap = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(
                    vmin=c_df.groupby("source_w_other")
                    .mean(numeric_only=True)["normalized value"]
                    .min()
                    * 0.8,
                    vmax=c_df.groupby("source_w_other")
                    .mean(numeric_only=True)["normalized value"]
                    .max(),
                ),
                cmap=curr_cmap,
            )
            cmaps[layer] = cmap
        colors = []

        new_df = pd.DataFrame()
        for source in sources:
            source_df = df[df["source_w_other"] == source].copy()
            if source <= other_id:
                colors.append("rgb(236,236,236, 0.75)")
                source_df["node_color"] = "rgb(236,236,236, 0.75)"
            elif source == root_id:
                colors.append("rgba(0,0,0,1)")
                source_df["node_color"] = "rgb(0,0,0,1)"
            else:
                intensity = (
                    source_df.groupby("source_w_other")
                    .mean(numeric_only=True)["normalized value"]
                    .values[0]
                )
                cmap = cmaps[source_df["source layer"].unique()[0]]
                r, g, b, a = cmap.to_rgba(intensity, alpha=0.75)
                colors.append(f"rgba({r * 255}, {g * 255}, {b * 255}, {a})")
                source_df["node_color"] = f"rgba({r * 255}, {g * 255}, {b * 255}, {a})"
            new_df = pd.concat([new_df, source_df])
        return new_df, colors

    def get_node_positions(feature_labels: list, df: pd.DataFrame):
        x = []
        y = []
        grouped_df = df.groupby("source_w_other", as_index=False).agg(
            {"source layer": "min", "value": "mean"}
        )
        layers = range(n_layers)
        final_df = pd.DataFrame()
        for layer in layers:
            layer_df = (
                grouped_df[grouped_df["source layer"] == layer]
                .sort_values(["value"], ascending=True)
                .copy()
            )
            other_df = layer_df[layer_df["source_w_other"] <= other_id].copy()
            layer_df = layer_df[layer_df["source_w_other"] > other_id].copy()

            other_value = other_df["value"]

            layer_df["rank"] = range(len(layer_df.index))
            layer_df["value"] = layer_df["value"] / layer_df["value"].sum()
            layer_df["y"] = (
                0.8
                * (0.01 + max(layer_df["rank"]) - layer_df["rank"])
                / (max(layer_df["rank"]))
                - 0.05
            )
            layer_df["x"] = (0.01 + layer) / (len(layers) + 1)
            other_df = pd.DataFrame(
                [
                    [
                        other_id * (layer + 1),
                        layer,
                        other_value,
                        10,
                        0.9,
                        (0.01 + layer) / (len(layers) + 1),
                    ]
                ],
                columns=["source_w_other", "source layer", "value", "rank", "y", "x"],
            )

            final_df = pd.concat([final_df, layer_df, other_df])

        for f in feature_labels:
            if f == root_id:
                x.append(0.85)
                y.append(0.5)
            else:
                x.append(final_df[final_df["source_w_other"] == f]["x"].values)
                y.append(final_df[final_df["source_w_other"] == f]["y"].values)
        return x, y

    df["source_w_other"] = df.apply(lambda x: set_to_other(x, top_n, "source"), axis=1)
    df["target_w_other"] = df.apply(lambda x: set_to_other(x, top_n, "target"), axis=1)
    df = df[
        ~((df["source_w_other"] <= other_id) & (df["target_w_other"] <= other_id))
    ].copy()
    df = normalize_layer_values(df)
    df = df.groupby(
        by=["source_w_other", "target_w_other", "type"], sort=False, as_index=False
    ).agg(
        {
            "normalized value": "sum",
            "value": "sum",
            "source layer": "mean",
            "target layer": "mean",
        }
    )
    unique_features = (
        df["source_w_other"].unique().tolist() + df["target_w_other"].unique().tolist()
    )
    code_map, feature_labels = _encode_features(list(set(unique_features)))
    sources = df["source_w_other"].unique().tolist()

    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(sources, df)
    x, y = get_node_positions(feature_labels, df)

    new_labels = []
    for label in feature_labels:
        if label in name_map.keys():
            new_labels.append(name_map[label])
        else:
            new_labels.append(label)

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
        source=encoded_source, target=encoded_target, value=value, color=link_colors
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


# ----------------------------------------------------------------


def _encode_features(features: list):
    feature_map = {"feature": features, "code": list(range(len(features)))}
    feature_map = pd.DataFrame(data=feature_map)
    return feature_map, features


def _get_code(feature: list, feature_map: pd.DataFrame):
    code = feature_map[feature_map["feature"] == feature]["code"].values[0]
    return code


def _remove_loops(df: pd.DataFrame):
    df.loc[:, "loop"] = df.apply(lambda x: x["source name"] == x["target name"], axis=1)
    df = df[df["loop"] == False].copy()
    return df


def _create_name_map(df: pd.DataFrame, n_layers: int, root_id: int, other_id: int):
    name_map = dict(
        zip(df["source"].values.tolist(), df["source name"].values.tolist())
    )
    name_map[root_id] = "Output"
    for i in range(1, n_layers + 1):
        name_map[other_id * i] = f"Other connections {i}"
    return name_map


def _create_subgraph_name_map(df: pd.DataFrame):
    name_map = dict(
        zip(df["source"].values.tolist(), df["source name"].values.tolist())
    )
    name_map.update(
        zip(df["target"].values.tolist(), df["target name"].values.tolist())
    )
    return name_map
