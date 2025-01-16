from typing import Union
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.graph_objects as go


class SankeyPlotter:
    """
    A utility class to plot a Plotly Sankey diagram from the output of BINNExplainer.
    """

    def __init__(
        self,
        explanations_data: pd.DataFrame,
        show_top_n: int = 10,
        value_col: str = "importance",
        node_cmap: str = "Reds",
        edge_cmap: Union[str, list] = "coolwarm",
    ):
        """
        :param explanations_data: DataFrame containing edge data.
        :param multiclass: Whether to group edges by class_idx or sum them all.
        :param show_top_n: How many top nodes per layer to keep. The rest become "Other".
        :param value_col: Name of the column that holds the numeric weight/importance.
        :param node_cmap: Matplotlib colormap name for node coloring.
        :param edge_cmap: Matplotlib colormap name (str) or a list of colors for edges in multiclass mode.
        """
        self.explanations_data = explanations_data.copy()
        self.show_top_n = show_top_n
        self.value_col = value_col
        self.node_cmap = node_cmap
        self.edge_cmap = edge_cmap

        # ID prefix for "other" nodes
        self.other_id = "nOther"

    def plot(self) -> go.Figure:
        """
        Main method to produce the Sankey figure.
        Returns:
            A Plotly Figure (go.Figure) containing the Sankey diagram.
        """
        # --- Ensure we have unique IDs for source/target (if not provided or if we want to re-generate).
        self.explanations_data["source_id"] = [
            f"n{row['source_node']}_l{row['source_layer']}"
            for _, row in self.explanations_data.iterrows()
        ]
        self.explanations_data["target_id"] = [
            f"n{row['target_node']}_l{row['target_layer']}"
            for _, row in self.explanations_data.iterrows()
        ]

        # --- Convert layers to int, remove loops, set 'value' column
        self.explanations_data["source_layer"] = self.explanations_data[
            "source_layer"
        ].astype(int)
        self.explanations_data["target_layer"] = self.explanations_data[
            "target_layer"
        ].astype(int)
        self.explanations_data = self._remove_loops(self.explanations_data)
        self.explanations_data["importance_val"] = self.explanations_data[
            self.value_col
        ]

        # --- Determine how many layers we have
        self.n_layers = self.explanations_data["target_layer"].max()

        # --- Create a map for naming display labels
        display_name_map = self._create_display_name_map(self.explanations_data)

        # --- Figure out top_n nodes per layer (for collapsing into "Other")
        top_n_map = {}
        for layer_idx in range(self.n_layers):
            layer_slice = self.explanations_data[
                self.explanations_data["source_layer"] == layer_idx
            ]
            layer_top_nodes = (
                layer_slice.groupby("source_id", as_index=True)[["importance_val"]]
                .mean(numeric_only=True)
                .sort_values("importance_val", ascending=False)
                .head(self.show_top_n)
                .index.tolist()
            )
            top_n_map[layer_idx] = layer_top_nodes

        # --- Rename out-of-top-n nodes as "nOther_l<layer>"
        self.explanations_data["source_w_other"] = self.explanations_data.apply(
            lambda row: self._set_to_other(row, top_n_map, "source_id"),
            axis=1,
        )
        self.explanations_data["target_w_other"] = self.explanations_data.apply(
            lambda row: self._set_to_other(row, top_n_map, "target_id"),
            axis=1,
        )

        # --- Normalize the values layer by layer, scaling "Other" differently
        self.explanations_data = self._normalize_layer_values(self.explanations_data)

        # --- Sum again by (source_w_other, target_w_other, class_idx)
        self.explanations_data = self.explanations_data.groupby(
            ["source_w_other", "target_w_other", "class_idx"],
            sort=False,
            as_index=False,
        ).agg(
            {
                "normalized_importance_val": "sum",
                "importance_val": "sum",
                "source_layer": "mean",
                "target_layer": "mean",
            }
        )

        # --- Prepare the unique node list and encode them
        unique_nodes = pd.Series(
            pd.concat(
                [
                    self.explanations_data["source_w_other"],
                    self.explanations_data["target_w_other"],
                ]
            ).unique()
        ).tolist()
        code_map, node_labels = self._encode_nodes(unique_nodes)

        # --- Assign node colors
        _, node_colors = self._get_node_colors(
            node_labels, self.explanations_data
        )

        # --- Build link (source/target/value/color) arrays
        encoded_source, encoded_target, link_values, link_colors = (
            self._get_connections(
                node_labels,
                self.explanations_data,
                code_map,
                node_labels,
            )
        )

        # --- Compute node positions
        x_positions, y_positions = self._get_node_positions(
            node_labels, self.explanations_data
        )

        # --- Remap labels for display
        display_labels = [
            display_name_map[label] if label in display_name_map else label
            for label in node_labels
        ]

        # --- Build the final Sankey figure
        fig = go.Figure(
            data=[
                go.Sankey(
                    textfont=dict(size=15, family="Arial"),
                    orientation="h",
                    arrangement="snap",
                    node=dict(
                        pad=15,
                        thickness=15,
                        line=dict(color="white", width=0),
                        label=display_labels,
                        color=node_colors,
                        x=x_positions,
                        y=y_positions,
                    ),
                    link=dict(
                        source=encoded_source,
                        target=encoded_target,
                        value=link_values,
                        color=link_colors,
                    ),
                )
            ]
        )
        return fig

    def _remove_loops(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Drop any edges where source_node == target_node (self loops).
        """
        data_frame["loop"] = data_frame.apply(
            lambda x: x["source_node"] == x["target_node"], axis=1
        )
        data_frame = data_frame[~data_frame["loop"]].copy()
        data_frame.drop(columns=["loop"], inplace=True)
        return data_frame

    def _create_display_name_map(self, data_frame: pd.DataFrame) -> dict:
        """
        Build a map from raw IDs (like "nA0M8Q6_l0") to a more readable name
        (like "A0M8Q6"), plus special mapping for 'other' nodes
        (e.g. "nOther_l2" -> "Other connections 2").
        """
        name_map = dict(
            zip(data_frame["source_id"].tolist(), data_frame["source_node"].tolist())
        )
        # Also define the mapping for "nOther_lX" -> "Other connections X"
        for i in range(1, self.n_layers + 3):
            key = f"{self.other_id}_l{i}"
            name_map[key] = f"Other connections {i}"
        return name_map

    def _set_to_other(self, row: pd.Series, top_n_map: dict, src_or_tgt: str) -> str:
        """
        If the node is not in top_n for its layer, rename it as nOther_l<layer>.
        """
        node_id = row[src_or_tgt]
        # For the "target_id", its conceptual layer is one step further
        layer = row["source_layer"] + 1
        if src_or_tgt == "target_id":
            layer += 1

        # If in top_n for any layer or is an "output_node", keep it
        for layer_values in top_n_map.values():
            if node_id in layer_values:
                return node_id
        if "output_node" in node_id:
            return node_id

        # Otherwise rename to "nOther_l<layer>"
        return f"{self.other_id}_l{layer}"

    def _normalize_layer_values(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        For "other" nodes, scale the value differently (e.g., by 0.1) so they
        don't dominate the visualization.
        """
        df_copy = data_frame.copy()
        df_copy["is_other"] = df_copy["source_w_other"].apply(
            lambda x: x.startswith(self.other_id)
        )

        # Separate out other vs. non-other
        other_data = df_copy[df_copy["is_other"]].copy()
        main_data = df_copy[~df_copy["is_other"]].copy()

        # Normalize within each layer
        result_df = pd.DataFrame()
        for layer_val in main_data["source_layer"].unique():
            layer_slice = main_data[main_data["source_layer"] == layer_val].copy()
            layer_total = layer_slice["importance_val"].sum()
            if layer_total != 0:
                layer_slice["normalized_importance_val"] = (
                    layer_slice["importance_val"] / layer_total
                )
            else:
                layer_slice["normalized_importance_val"] = 0.0

            result_df = pd.concat([result_df, layer_slice], ignore_index=True)

        # Scale "other" rows differently (e.g. 0.1)
        for layer_val in other_data["source_layer"].unique():
            layer_slice = other_data[other_data["source_layer"] == layer_val].copy()
            layer_total = layer_slice["importance_val"].sum()
            if layer_total != 0:
                layer_slice["normalized_importance_val"] = (
                    0.1 * layer_slice["importance_val"] / layer_total
                )
            else:
                layer_slice["normalized_importance_val"] = 0.0

            result_df = pd.concat([result_df, layer_slice], ignore_index=True)

        return result_df

    def _encode_nodes(self, features: list):
        """
        Assign each unique feature string an integer code.
        """
        feature_map_df = pd.DataFrame(
            {"feature": features, "code": list(range(len(features)))}
        )
        return feature_map_df, features

    def _get_code(self, feature: str, feature_map_df: pd.DataFrame) -> int:
        """
        Look up the integer code for the given feature string.
        """
        return feature_map_df.loc[feature_map_df["feature"] == feature, "code"].values[
            0
        ]

    def _get_node_colors(self, node_list: list, data_frame: pd.DataFrame):
        """
        Color each node according to a colormap based on the node's
        average normalized_value. 'Other' nodes become light grey.
        """
        # Build a per-layer colormap
        unique_layers = data_frame["source_layer"].unique()
        layer_cmaps = {}
        for layer_val in unique_layers:
            layer_slice = data_frame[data_frame["source_layer"] == layer_val].copy()
            # only real nodes for min/max
            real_nodes_slice = layer_slice[
                ~layer_slice["source_w_other"].str.startswith(self.other_id)
            ]
            if not real_nodes_slice.empty:
                vmin = (
                    real_nodes_slice.groupby("source_w_other")[
                        "normalized_importance_val"
                    ]
                    .mean()
                    .min()
                )
                vmax = (
                    real_nodes_slice.groupby("source_w_other")[
                        "normalized_importance_val"
                    ]
                    .mean()
                    .max()
                )
            else:
                vmin, vmax = 0, 1
            mpl_cmap = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=vmin * 0.8, vmax=vmax),
                cmap=self.node_cmap,
            )
            layer_cmaps[layer_val] = mpl_cmap

        node_colors = []
        new_df = pd.DataFrame()
        for node_name in node_list:
            node_slice = data_frame[data_frame["source_w_other"] == node_name].copy()

            if node_name.startswith(self.other_id):
                # color "Other" nodes: light grey
                color_str = "rgba(236,236,236,0.75)"
                node_colors.append(color_str)
                node_slice["node_color"] = color_str
            elif "output_node" in node_name:
                # black for an output node?
                color_str = "rgba(0,0,0,1)"
                node_colors.append(color_str)
                node_slice["node_color"] = color_str
            else:
                # normal node
                avg_intensity = node_slice["normalized_importance_val"].mean()
                layer_of_node = int(node_slice["source_layer"].mean())
                mpl_cmap = layer_cmaps.get(layer_of_node)
                if mpl_cmap is None:
                    color_str = "rgba(200,200,200,0.75)"  # fallback
                else:
                    r, g, b, a = mpl_cmap.to_rgba(avg_intensity, alpha=0.75)
                    color_str = f"rgba({r*255:.0f},{g*255:.0f},{b*255:.0f},{a:.2f})"
                node_colors.append(color_str)
                node_slice["node_color"] = color_str

            new_df = pd.concat([new_df, node_slice], ignore_index=True)

        return new_df, node_colors

    def _get_connections(
        self,
        node_list: list,
        data_frame: pd.DataFrame,
        code_map: pd.DataFrame,
        all_node_labels: list,
    ):
        """
        Build up source, target, value, and link_color arrays to feed Plotly Sankey.
        """
        # We only need the subset that matches these node_list
        subset_df = data_frame[data_frame["source_w_other"].isin(node_list)].copy()

        source_codes = [
            self._get_code(s, code_map) for s in subset_df["source_w_other"]
        ]
        target_codes = [
            self._get_code(t, code_map) for t in subset_df["target_w_other"]
        ]
        values = subset_df["normalized_importance_val"].tolist()

        # Single color scale:
        temp_df, _ = self._get_node_colors(
            all_node_labels, data_frame
        )
        # Quick hack: pick one color per row (matching node color).
        link_colors = (
            temp_df["node_color"]
            .apply(lambda x: x.split(",0.75)")[0] + ",0.75)")
            .tolist()
        )

        return source_codes, target_codes, values, link_colors

    def _get_node_positions(
        self, node_labels: list, df: pd.DataFrame
    ):
        """
        For each layer, compute x,y positions so that 'Other' lumps get placed
        at the bottom, and real nodes get spread top-to-bottom.

        We rely on 'source_layer' or 'target_layer' to place the node. 
        We'll group by node_w_other to find out the node's final layer 
        and average importance, then place them on the Sankey.
        """
        summary = []
        for node_id in node_labels:
            if "output_node" in node_id:
                relevant = df[
                    (df["target_w_other"] == node_id)
                ]
                layer = int(min(relevant["target_layer"]))
            else:
                relevant = df[
                    (df["source_w_other"] == node_id)
                ]
                layer = int(min(relevant["source_layer"]))
            
            avg_val = relevant["importance_val"].mean()
            is_other = node_id.startswith(self.other_id)
            summary.append(
                dict(node_id=node_id, layer=layer, avg_value=avg_val, is_other=is_other)
            )

        pos_df = pd.DataFrame(summary)

        final_positions = []
        for layer_idx in range(self.n_layers):
            sub = pos_df[pos_df["layer"] == layer_idx].copy()
            # separate real vs other
            other_sub = sub[sub["is_other"]]
            real_sub = sub[~sub["is_other"]]

            # sort real nodes ascending
            real_sub = real_sub.sort_values("avg_value", ascending=True).reset_index(drop=True)
            real_sub["rank"] = real_sub.index

            # if sum of values is zero, skip dividing
            if real_sub["avg_value"].sum() > 0:
                real_sub["norm_val"] = real_sub["avg_value"] / real_sub["avg_value"].sum()
            else:
                real_sub["norm_val"] = 0.0

            # Spread them from [0.0..0.8]
            max_rank = real_sub["rank"].max() if len(real_sub) > 0 else 0
            if max_rank > 0:
                real_sub["y"] = 0.8 * (max_rank - real_sub["rank"]) / (max_rank + 0.001)
            else:
                real_sub["y"] = 0.4  # if there's just one node, put it in the middle

            real_sub["x"] = (0.02 + layer_idx) / (self.n_layers + 0.5)

            # place the other node(s) at y=0.95
            for idx in other_sub.index:
                other_node = other_sub.loc[idx, "node_id"]
                final_positions.append(
                    dict(
                        node_id=other_node,
                        x=(0.02 + layer_idx) / (self.n_layers + 0.5),
                        y=0.95,  # near bottom
                    )
                )

            # gather real_sub
            for idx in real_sub.index:
                final_positions.append(
                    dict(
                        node_id=real_sub.loc[idx, "node_id"],
                        x=real_sub.loc[idx, "x"],
                        y=real_sub.loc[idx, "y"],
                    )
                )

        # Build x,y arrays in node_labels order
        x_positions, y_positions = [], []
        pos_df2 = pd.DataFrame(final_positions)
        print(pos_df2)
        for label in node_labels:
            if "output_node" in label:
                x_positions.append(.85)
                y_positions.append(.5)
            else:
                row = pos_df2[pos_df2["node_id"] == label]
                x_positions.append(row["x"].values[0])
                y_positions.append(row["y"].values[0])

        return x_positions, y_positions
