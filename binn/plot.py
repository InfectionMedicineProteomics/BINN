import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.graph_objects as go
import shap


def encode_features(features):
    feature_map = {"feature": features, "code": list(range(len(features)))}
    feature_map = pd.DataFrame(data=feature_map)
    return feature_map, features


def get_code(feature, feature_map):
    code = feature_map[feature_map["feature"] == feature]["code"].values[0]
    return code


def remove_loops(df):
    df["loop"] = df.apply(lambda x: x["source"] == x["target"], axis=1).copy()
    df = df[df["loop"] == False]
    return df


def shap_sankey(
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
    code_map, feature_labels = encode_features(list(set(unique_features)))
    sources = df["source"].unique().tolist()

    def normalize_layer_values(df):
        new_df = pd.DataFrame()
        total_value_sum = df[val_col].sum()
        for layer in df["source layer"].unique():
            layer_df = df.loc[df["source layer"] == layer].copy()
            layer_total = layer_df[val_col].sum()
            layer_df["normalized value"] = (
                total_value_sum * layer_df[val_col] / layer_total
            )
            new_df = pd.concat([new_df, layer_df])
        return new_df

    df = remove_loops(df)
    df = normalize_layer_values(df)

    def get_connections(sources, source_target_df):
        conn = source_target_df[source_target_df["source"].isin(sources)]
        source_code = [get_code(s, code_map) for s in conn["source"]]
        target_code = [get_code(s, code_map) for s in conn["target"]]
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
            w = df[df["source layer"] == layer]["normalized value"].values
            n = df[df["source layer"] == layer]["source"].values

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
        new_df["node_color"] = [node_dict[n] for n in new_df["source"]]
        new_df["node_weight"] = [weight_dict[n] for n in new_df["source"]]
        return new_df, colors

    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(
        sources, df)
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


def complete_shap_sankey(
    df,
    show_top_n=10,
    savename="sepsis_complete_sankey",
    val_col="value",
    cmap_name="Reds",
):
    """
    Generate a Sankey diagram to visualize feature contributions using SHAP values.

    Parameters:
        df (pandas.DataFrame): A DataFrame with columns "source", "target", "source layer", and "target layer".
        show_top_n (int, optional): The number of top features to show in each layer. Default is 10.
        savename (str, optional): The name to use when saving the Sankey diagram.
        val_col (str, optional): The name of the column in the input DataFrame that contains the values to use for the width of the connections. Default is "value".
        cmap_name (str, optional): The name of the matplotlib colormap to use for coloring the links between the nodes. Default is "Reds".

    Returns:
        fig : go.Figure
            The Sankey diagram figure created using Plotly.
    """

    df["source layer"] = df["source layer"].astype(int).copy()
    df["target layer"] = df["target layer"].astype(int).copy()
    df = remove_loops(df)
    n_layers = max(df["target layer"].values)
    df["value"] = df[val_col]

    def get_top_n(df, layer, n):
        l = df.loc[df["source layer"] == layer]
        s = (
            l.groupby("source", as_index=False)
            .mean()
            .sort_values("value", ascending=False)[0:n]
        )
        top_n_source = s["source"].values.tolist()
        return top_n_source

    top_n = {}
    for layer in range(n_layers):
        if layer < n_layers:
            top_n[layer] = get_top_n(df, layer, show_top_n)

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
        def contains_other(x):
            if ("Other" in x["source_w_other"]) and ("Other" in x["target_w_other"]):
                return True
            elif ("Other" in x["source_w_other"]) and ("root" in x["target_w_other"]):
                return False  # can change to True if want to remove Other to root
            return False

        df["is_other_to_other"] = df.apply(
            lambda x: contains_other(x), axis=1).copy()
        df = df[df["is_other_to_other"] == False]
        return df

    df["source_w_other"] = df.apply(
        lambda x: set_to_other(x, top_n, "source"), axis=1
    ).copy()
    df["target_w_other"] = df.apply(
        lambda x: set_to_other(x, top_n, "target"), axis=1
    ).copy()

    df = remove_other_to_other(df)

    def normalize_layer_values(df):
        new_df = pd.DataFrame()
        df["Other"] = (
            df["source"]
            .apply(lambda x: True if "Other connections" in x else False)
            .copy()
        )
        other_df = df[df["Other"] == True]
        df = df[df["Other"] == False]
        for layer in df["source layer"].unique():
            layer_df = df.loc[df["source layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = 1 * layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        for layer in other_df["source layer"].unique():
            layer_df = other_df.loc[df["source layer"] == layer]
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = 0.1 * \
                layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        return new_df

    df = normalize_layer_values(df)

    unique_features = df["source_w_other"].unique().tolist()
    unique_features += df["target_w_other"].unique().tolist()
    code_map, feature_labels = encode_features(list(set(unique_features)))
    sources = df["source_w_other"].unique().tolist()

    def get_connections(sources, source_target_df):
        conn = source_target_df[source_target_df["source_w_other"].isin(
            sources)]
        source_code = [get_code(s, code_map) for s in conn["source_w_other"]]
        target_code = [get_code(s, code_map) for s in conn["target_w_other"]]
        values = [v for v in conn["normalized value"]]
        link_colors = (
            conn["node_color"]
            .apply(lambda x: x.split(", 0.75)")[0] + ", 0.3)")
            .values.tolist()
        )
        return source_code, target_code, values, link_colors

    def get_node_colors(sources, df):
        cmaps = {}
        for layer in df["source layer"].unique():
            c_df = df[df["source layer"] == layer]
            c_df = c_df[~c_df["source_w_other"].str.startswith("Other")].copy()
            max_value = c_df.groupby("source_w_other").mean()[
                "normalized value"].max()
            min_value = (
                c_df.groupby("source_w_other").mean()[
                    "normalized value"].min() * 0.8
            )
            cmap = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(
                    vmin=min_value, vmax=max_value),
                cmap=cmap_name,
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
            layer_df = layer_df.loc[~layer_df["source_w_other"].str.startswith(
                "Other")]
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
                columns=["source_w_other", "source layer",
                         "value", "rank", "y", "x"],
            )
            final_df = pd.concat([final_df, layer_df, other_df])

        for f in feature_labels:
            if f == "root":
                x.append(0.85)
                y.append(0.5)
            else:
                x.append(
                    final_df[final_df["source_w_other"] == f]["x"].values[0])
                y.append(
                    final_df[final_df["source_w_other"] == f]["y"].values[0])
        return x, y

    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(
        sources, df)
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
    fig.write_image(f"{savename}", width=1900, scale=2, height=800)
    return fig


def plot_shap_summary(shap_values, test_data, feature_names, output_path):

    shap.summary_plot(
        shap_values,
        test_data,
        feature_names=feature_names,
        max_display=30,
        plot_size=[15, 6],
    )
    plt.savefig(output_path, dpi=200)


def multiclass_shap_sankey(
    df,
    show_top_n=10,
    savename="sepsis_complete_sankey",
    val_col="value",
    color_list:list = ['rgba(255,0,0,1)','rgba(0,255,0,1)','rgba(0,0,255,1)'],
    node_cmap='Reds'
):
    """
    Generate a Sankey diagram to visualize feature contributions using SHAP values.

    Parameters:
        df (pandas.DataFrame): A DataFrame with columns "source", "target", "source layer", and "target layer".
        show_top_n (int, optional): The number of top features to show in each layer. Default is 10.
        savename (str, optional): The name to use when saving the Sankey diagram.
        val_col (str, optional): The name of the column in the input DataFrame that contains the values to use for the width of the connections. Default is "value".
        cmap_name (str, optional): The name of the matplotlib colormap to use for coloring the links between the nodes. Default is "Reds".

    Returns:
        fig : go.Figure
            The Sankey diagram figure created using Plotly.
    """

    df["source layer"] = df["source layer"].astype(int).copy()
    df["target layer"] = df["target layer"].astype(int).copy()
    df = remove_loops(df)
    n_layers = max(df["target layer"].values)
    df["value"] = df[val_col]

    def get_top_n(df, layer, n):
        l = df.loc[df["source layer"] == layer]
        s = (
            l.groupby("source", as_index=False)
            .mean()
            .sort_values("value", ascending=False)[0:n]
        )
        top_n_source = s["source"].values.tolist()
        return top_n_source

    top_n = {}
    for layer in range(n_layers):
        if layer < n_layers:
            top_n[layer] = get_top_n(df, layer, show_top_n)

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
        def contains_other(x):
            if ("Other" in x["source_w_other"]) and ("Other" in x["target_w_other"]):
                return True
            elif ("Other" in x["source_w_other"]) and ("root" in x["target_w_other"]):
                return False  # can change to True if want to remove Other to root
            return False

        df["is_other_to_other"] = df.apply(
            lambda x: contains_other(x), axis=1).copy()
        df = df[df["is_other_to_other"] == False]
        return df

    df["source_w_other"] = df.apply(
        lambda x: set_to_other(x, top_n, "source"), axis=1
    ).copy()
    df["target_w_other"] = df.apply(
        lambda x: set_to_other(x, top_n, "target"), axis=1
    ).copy()

    df = remove_other_to_other(df)

    def normalize_layer_values(df):
        new_df = pd.DataFrame()
        df["Other"] = (
            df["source"]
            .apply(lambda x: True if "Other connections" in x else False)
            .copy()
        )
        other_df = df[df["Other"] == True]
        df = df[df["Other"] == False]
        for layer in df["source layer"].unique():
            layer_df = df.loc[df["source layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = 1 * layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        for layer in other_df["source layer"].unique():
            layer_df = other_df.loc[df["source layer"] == layer]
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = 0.1 * \
                layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        return new_df

    df = normalize_layer_values(df)

    unique_features = df["source_w_other"].unique().tolist()
    unique_features += df["target_w_other"].unique().tolist()
    code_map, feature_labels = encode_features(list(set(unique_features)))
    sources = df["source_w_other"].unique().tolist()

    def get_connections(sources, source_target_df):
        conn = source_target_df[source_target_df["source_w_other"].isin(
            sources)]
        source_code = [get_code(s, code_map) for s in conn["source_w_other"]]
        target_code = [get_code(s, code_map) for s in conn["target_w_other"]]
        values = [v for v in conn["normalized value"]]

        link_colors = (
            conn["type"]
            .apply(lambda x: color_list[x])
            .values.tolist()
        )
        return source_code, target_code, values, link_colors

    def get_node_colors(sources, df):
        cmaps = {}
        for layer in df["source layer"].unique():
            c_df = df[df["source layer"] == layer]
            c_df = c_df[~c_df["source_w_other"].str.startswith("Other")].copy()
            max_value = c_df.groupby("source_w_other").mean()[
                "normalized value"].max()
            min_value = (
                c_df.groupby("source_w_other").mean()[
                    "normalized value"].min() * 0.8
            )
            cmap = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(
                    vmin=min_value, vmax=max_value),
                cmap=node_cmap,
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
            layer_df = layer_df.loc[~layer_df["source_w_other"].str.startswith(
                "Other")]
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
                columns=["source_w_other", "source layer",
                         "value", "rank", "y", "x"],
            )
            final_df = pd.concat([final_df, layer_df, other_df])

        for f in feature_labels:
            if f == "root":
                x.append(0.85)
                y.append(0.5)
            else:
                x.append(
                    final_df[final_df["source_w_other"] == f]["x"].values[0])
                y.append(
                    final_df[final_df["source_w_other"] == f]["y"].values[0])
        return x, y

    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(
        sources, df)
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
    fig.write_image(f"{savename}", width=1900, scale=2, height=800)
    return fig