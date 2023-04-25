from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.graph_objects as go


def subgraph_sankey(
    df: pd.DataFrame, final_node: str = "root", val_col="value", cmap_name="coolwarm"
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
    df["source layer"] = df["source layer"].astype(int).copy()
    df["target layer"] = df["target layer"].astype(int).copy()
    unique_features = df["source"].unique().tolist()
    unique_features += df["target"].unique().tolist()
    code_map, feature_labels = _encode_features(list(set(unique_features)))
    sources = df["source"].unique().tolist()

    def normalize_layer_values(df):
        new_df = pd.DataFrame()
        total_value_sum = df[val_col].sum()
        for layer in df["source layer"].unique():
            layer_df = df.loc[df["source layer"] == layer].loc[:, :].copy()
            layer_total = layer_df[val_col].sum()
            layer_df.loc[:, "normalized value"] = (
                total_value_sum * layer_df[val_col] / layer_total
            )
            new_df = pd.concat([new_df, layer_df])
        return new_df

    df = _remove_loops(df)
    df = normalize_layer_values(df)

    def get_connections(sources, source_target_df):
        conn = source_target_df[source_target_df["source"].isin(sources)]
        source_code = [_get_code(s, code_map) for s in conn["source"]]
        target_code = [_get_code(s, code_map) for s in conn["target"]]
        values = [v for v in conn["normalized value"]]
        link_colors = conn["node_color"].values.tolist()
        return source_code, target_code, values, link_colors

    def get_node_colors(sources, df):
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
        node_dict[final_node] = f"rgba(0,0,0,1)"
        weight_dict[final_node] = 1
        colors = [node_dict[n] for n in sources]
        new_df = new_df.assign(
            node_color=[node_dict[n] for n in new_df["source"]],
            node_weight=[weight_dict[n] for n in new_df["source"]],
        )
        return new_df, colors

    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(sources, df)
    nodes = dict(
        pad=20,
        thickness=20,
        line=dict(color="white", width=2),
        label=feature_labels,
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
    edge_cmap: Union[str, list] = "Reds",
):

    df["source layer"] = df["source layer"].astype(int).copy()
    df["target layer"] = df["target layer"].astype(int).copy()
    df = _remove_loops(df)
    n_layers = max(df["target layer"].values)
    df["value"] = df[val_col]

    top_n = {}
    for layer in range(n_layers):
        top_n[layer] = (
            df.loc[df["source layer"] == layer]
            .groupby("source")
            .mean()
            .sort_values("value", ascending=False)
            .iloc[:show_top_n]
            .index.tolist()
        )

    def set_to_other(row, top_n, source_or_target):
        s = row[source_or_target]
        layer = row["source layer"]
        if source_or_target == "target":
            layer = layer + 1
        for t in top_n.values():
            if s in t:
                return s
            if "root" in s:
                return "root"
        return f"Other connections {layer}"

    def remove_other_to_other(df):
        df["source_w_other"] = df.apply(
            lambda x: set_to_other(x, top_n, "source"), axis=1
        )
        df["target_w_other"] = df.apply(
            lambda x: set_to_other(x, top_n, "target"), axis=1
        )
        return df[
            ~(
                (df["source_w_other"].str.contains("Other connections"))
                & (df["target_w_other"].str.contains("Other connections"))
            )
        ]

    def normalize_layer_values(df):
        new_df = pd.DataFrame()
        df["Other"] = (
            df["source_w_other"]
            .apply(lambda x: True if "Other connections" in x else False)
            .copy()
        )
        other_df = df[df["Other"] == True]
        df = df[df["Other"] == False]
        for layer in df["source layer"].unique():
            layer_df = df.loc[df["source layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        for layer in other_df["source layer"].unique():
            layer_df = other_df.loc[other_df["source layer"] == layer]
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = 0.1 * layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        return new_df

    def get_connections(sources, df):
        conn = df[df["source_w_other"].isin(sources)]
        source_code = [_get_code(s, code_map) for s in conn["source_w_other"]]
        target_code = [_get_code(s, code_map) for s in conn["target_w_other"]]
        values = [v for v in conn["normalized value"]]
        if multiclass == False:
            temp_df, _ = get_node_colors(feature_labels, df, curr_cmap=edge_cmap)
            link_colors = (
                temp_df["node_color"]
                .apply(lambda x: x.split(", 0.75)")[0] + ", 0.3)")
                .values.tolist()
            )
        else:
            link_colors = conn.apply(
                lambda x: "rgba(236,236,236, 0.75)"
                if "Other connections" in x["source_w_other"]
                else edge_cmap[x["type"]],
                axis=1,
            ).values.tolist()
        return source_code, target_code, values, link_colors

    def get_node_colors(sources, df, curr_cmap=node_cmap):
        cmaps = {}
        for layer in df["source layer"].unique():
            c_df = df[df["source layer"] == layer]
            c_df = c_df[~c_df["source_w_other"].str.startswith("Other")].copy()
            cmap = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(
                    vmin=c_df.groupby("source_w_other").mean()["normalized value"].min()
                    * 0.8,
                    vmax=c_df.groupby("source_w_other")
                    .mean()["normalized value"]
                    .max(),
                ),
                cmap=curr_cmap,
            )
            cmaps[layer] = cmap
        colors = []

        new_df = pd.DataFrame()
        for source in sources:
            source_df = df[df["source_w_other"] == source]
            if "Other connections" in source:
                colors.append("rgb(236,236,236, 0.75)")
                source_df["node_color"] = "rgb(236,236,236, 0.75)"
            elif "root" in source:
                colors.append("rgba(0,0,0,1)")
                source_df["node_color"] = "rgb(0,0,0,1)"
            else:
                intensity = (
                    source_df.groupby("source_w_other")
                    .mean()["normalized value"]
                    .values[0]
                )
                cmap = cmaps[source_df["source layer"].unique()[0]]
                r, g, b, a = cmap.to_rgba(intensity, alpha=0.75)
                colors.append(f"rgba({r * 255}, {g * 255}, {b * 255}, {a})")
                source_df["node_color"] = f"rgba({r * 255}, {g * 255}, {b * 255}, {a})"
            new_df = pd.concat([new_df, source_df])
        return new_df, colors

    def get_node_positions(feature_labels, df):
        x = []
        y = []
        grouped_df = df.groupby("source_w_other", as_index=False).agg(
            {"source layer": "min", "value": "mean"}
        )
        layers = range(n_layers)
        final_df = pd.DataFrame()
        for layer in layers:
            other_df = grouped_df.loc[
                grouped_df["source_w_other"].str.startswith("Other")
            ]
            other_value = other_df.groupby("source_w_other").mean().value[0]
            layer_df = grouped_df[grouped_df["source layer"] == layer].sort_values(
                ["value"], ascending=True
            )
            layer_df = layer_df.loc[~layer_df["source_w_other"].str.startswith("Other")]
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
                        f"Other connections {layer}",
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
            if f == "root":
                x.append(0.85)
                y.append(0.5)
            else:
                x.append(final_df[final_df["source_w_other"] == f]["x"].values[0])
                y.append(final_df[final_df["source_w_other"] == f]["y"].values[0])
        return x, y

    df = remove_other_to_other(df)
    df = normalize_layer_values(df)
    unique_features = (
        df["source_w_other"].unique().tolist() + df["target_w_other"].unique().tolist()
    )
    code_map, feature_labels = _encode_features(list(set(unique_features)))
    sources = df["source_w_other"].unique().tolist()
    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(sources, df)
    x, y = get_node_positions(feature_labels, df)

    feature_labels = [f.split("_")[0] for f in feature_labels]
    nodes = dict(
        pad=15,
        thickness=15,
        line=dict(color="white", width=0),
        label=feature_labels,
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


def _encode_features(features):
    feature_map = {"feature": features, "code": list(range(len(features)))}
    feature_map = pd.DataFrame(data=feature_map)
    return feature_map, features


def _get_code(feature, feature_map):
    code = feature_map[feature_map["feature"] == feature]["code"].values[0]
    return code


def _remove_loops(df):
    df.loc[:, "loop"] = df.apply(lambda x: x["source"] == x["target"], axis=1)
    df = df[df["loop"] == False]
    return df
