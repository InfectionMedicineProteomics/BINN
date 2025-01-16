from typing import Union
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.graph_objects as go

class SankeyPlotter:
    """
    A utility class to plot a Plotly Sankey diagram from a DataFrame 
    describing source-to-target connections (edges).

    Example DataFrame columns (one row per edge):
      source_layer, target_layer, source_node, target_node, class_idx, importance, 
      source_id, target_id, normalized_importance
    """

    def __init__(
        self,
        explanations_data: pd.DataFrame,
        multiclass: bool = False,
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
        self.multiclass = multiclass
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

        # --- If we're not in multiclass mode, group edges accordingly.
        if not self.multiclass:
            self.explanations_data = (
                self.explanations_data.groupby(
                    ["source_id", "target_id", "source_node", "target_node"],
                    as_index=False,
                )
                .agg(
                    {
                        self.value_col: "sum",
                        "source_layer": "mean",
                        "target_layer": "mean",
                        "class_idx": "mean",
                    }
                )
            )

        # --- Convert layers to int, remove loops, set 'value' column
        self.explanations_data["source_layer"] = self.explanations_data["source_layer"].astype(int)
        self.explanations_data["target_layer"] = self.explanations_data["target_layer"].astype(int)
        self.explanations_data = self._remove_loops(self.explanations_data)
        self.explanations_data["value"] = self.explanations_data[self.value_col]

        # --- Determine how many layers we have
        n_layers = self.explanations_data["target_layer"].max()

        # --- Create a map for naming display labels
        display_name_map = self._create_display_name_map(
            self.explanations_data, n_layers, self.other_id
        )

        # --- Figure out top_n nodes per layer (for collapsing into "Other")
        top_n_map = {}
        for layer_idx in range(n_layers):
            layer_slice = self.explanations_data[
                self.explanations_data["source_layer"] == layer_idx
            ]
            layer_top_nodes = (
                layer_slice.groupby("source_id", as_index=True)[["value"]]
                .mean(numeric_only=True)
                .sort_values("value", ascending=False)
                .head(self.show_top_n)
                .index.tolist()
            )
            top_n_map[layer_idx] = layer_top_nodes

        # --- Rename out-of-top-n nodes as "nOther_l<layer>"
        self.explanations_data["source_w_other"] = self.explanations_data.apply(
            lambda row: self._set_to_other(row, top_n_map, "source_id", self.other_id),
            axis=1,
        )
        self.explanations_data["target_w_other"] = self.explanations_data.apply(
            lambda row: self._set_to_other(row, top_n_map, "target_id", self.other_id),
            axis=1,
        )

        # --- Normalize the values layer by layer, scaling "Other" differently
        self.explanations_data = self._normalize_layer_values(
            self.explanations_data, self.other_id
        )

        # --- Sum again by (source_w_other, target_w_other, class_idx)
        self.explanations_data = (
            self.explanations_data.groupby(
                ["source_w_other", "target_w_other", "class_idx"],
                sort=False,
                as_index=False,
            )
            .agg(
                {
                    "normalized_value": "sum",
                    "value": "sum",
                    "source_layer": "mean",
                    "target_layer": "mean",
                }
            )
        )

        # --- Prepare the unique node list and encode them
        unique_nodes = (
            pd.Series(
                pd.concat(
                    [
                        self.explanations_data["source_w_other"],
                        self.explanations_data["target_w_other"],
                    ]
                ).unique()
            ).tolist()
        )
        code_map, node_labels = self._encode_nodes(unique_nodes)

        # --- Assign node colors
        _, node_colors = self._get_node_colors(
            node_labels, self.explanations_data, self.node_cmap, self.other_id
        )

        # --- Build link (source/target/value/color) arrays
        encoded_source, encoded_target, link_values, link_colors = self._get_connections(
            node_labels, self.explanations_data, code_map, self.multiclass,
            self.edge_cmap, node_labels, self.other_id
        )

        # --- Compute node positions
        x_positions, y_positions = self._get_node_positions(
            node_labels, self.explanations_data, n_layers, self.other_id
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

    # ------------------------------------------------------------------------
    # Below are private/static helper methods
    # ------------------------------------------------------------------------

    @staticmethod
    def _remove_loops(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Drop any edges where source_node == target_node (self loops).
        """
        data_frame["loop"] = data_frame.apply(
            lambda x: x["source_node"] == x["target_node"], axis=1
        )
        data_frame = data_frame[~data_frame["loop"]].copy()
        data_frame.drop(columns=["loop"], inplace=True)
        return data_frame

    @staticmethod
    def _create_display_name_map(
        data_frame: pd.DataFrame, n_layers: int, other_id: str
    ) -> dict:
        """
        Build a map from raw IDs (like "nA0M8Q6_l0") to a more readable name 
        (like "A0M8Q6"), plus special mapping for 'other' nodes 
        (e.g. "nOther_l2" -> "Other connections 2").
        """
        name_map = dict(
            zip(data_frame["source_id"].tolist(), data_frame["source_node"].tolist())
        )
        # Also define the mapping for "nOther_lX" -> "Other connections X"
        for i in range(1, n_layers + 3):
            key = f"{other_id}_l{i}"
            name_map[key] = f"Other connections {i}"
        return name_map

    @staticmethod
    def _set_to_other(row: pd.Series, top_n_map: dict, src_or_tgt: str, other_id: str) -> str:
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
        return f"{other_id}_l{layer}"

    @staticmethod
    def _normalize_layer_values(data_frame: pd.DataFrame, other_id: str) -> pd.DataFrame:
        """
        For "other" nodes, scale the value differently (e.g., by 0.1) so they 
        don't dominate the visualization.
        """
        df_copy = data_frame.copy()
        df_copy["is_other"] = df_copy["source_w_other"].apply(
            lambda x: x.startswith(other_id)
        )

        # Separate out other vs. non-other
        other_data = df_copy[df_copy["is_other"]].copy()
        main_data = df_copy[~df_copy["is_other"]].copy()

        # Normalize within each layer
        result_df = pd.DataFrame()
        for layer_val in main_data["source_layer"].unique():
            layer_slice = main_data[main_data["source_layer"] == layer_val].copy()
            layer_total = layer_slice["value"].sum()
            if layer_total != 0:
                layer_slice["normalized_value"] = layer_slice["value"] / layer_total
            else:
                layer_slice["normalized_value"] = 0.0

            result_df = pd.concat([result_df, layer_slice], ignore_index=True)

        # Scale "other" rows differently (e.g. 0.1)
        for layer_val in other_data["source_layer"].unique():
            layer_slice = other_data[other_data["source_layer"] == layer_val].copy()
            layer_total = layer_slice["value"].sum()
            if layer_total != 0:
                layer_slice["normalized_value"] = 0.1 * layer_slice["value"] / layer_total
            else:
                layer_slice["normalized_value"] = 0.0

            result_df = pd.concat([result_df, layer_slice], ignore_index=True)

        return result_df

    @staticmethod
    def _encode_nodes(features: list):
        """
        Assign each unique feature string an integer code.
        """
        feature_map_df = pd.DataFrame(
            {"feature": features, "code": list(range(len(features)))}
        )
        return feature_map_df, features

    @staticmethod
    def _get_code(feature: str, feature_map_df: pd.DataFrame) -> int:
        """
        Look up the integer code for the given feature string.
        """
        return feature_map_df.loc[feature_map_df["feature"] == feature, "code"].values[0]

    @staticmethod
    def _get_node_colors(
        node_list: list, data_frame: pd.DataFrame, cmap_name, other_id
    ):
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
                ~layer_slice["source_w_other"].str.startswith(other_id)
            ]
            if not real_nodes_slice.empty:
                vmin = real_nodes_slice.groupby("source_w_other")["normalized_value"].mean().min()
                vmax = real_nodes_slice.groupby("source_w_other")["normalized_value"].mean().max()
            else:
                vmin, vmax = 0, 1
            mpl_cmap = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=vmin * 0.8, vmax=vmax),
                cmap=cmap_name,
            )
            layer_cmaps[layer_val] = mpl_cmap

        node_colors = []
        new_df = pd.DataFrame()
        for node_name in node_list:
            node_slice = data_frame[data_frame["source_w_other"] == node_name].copy()

            if node_name.startswith(other_id):
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
                avg_intensity = node_slice["normalized_value"].mean()
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

    @staticmethod
    def _get_connections(
        node_list: list,
        data_frame: pd.DataFrame,
        code_map: pd.DataFrame,
        multiclass: bool,
        edge_cmap,
        all_node_labels: list,
        other_id: str,
    ):
        """
        Build up source, target, value, and link_color arrays to feed Plotly Sankey.
        """
        # We only need the subset that matches these node_list
        subset_df = data_frame[data_frame["source_w_other"].isin(node_list)].copy()

        source_codes = [
            SankeyPlotter._get_code(s, code_map) for s in subset_df["source_w_other"]
        ]
        target_codes = [
            SankeyPlotter._get_code(t, code_map) for t in subset_df["target_w_other"]
        ]
        values = subset_df["normalized_value"].tolist()

        if not multiclass:
            # Single color scale:
            temp_df, _ = SankeyPlotter._get_node_colors(
                all_node_labels, data_frame, edge_cmap, other_id
            )
            # Quick hack: pick one color per row (matching node color).
            link_colors = (
                temp_df["node_color"]
                .apply(lambda x: x.split(",0.75)")[0] + ",0.75)")
                .tolist()
            )
        else:
            # If multiclass, maybe color by class index from a list of colors
            link_colors = subset_df.apply(
                lambda x: (
                    "rgba(236,236,236,0.75)"
                    if x["source_w_other"].startswith(other_id)
                    else edge_cmap[int(x["class_idx"]) % len(edge_cmap)]
                ),
                axis=1,
            ).tolist()

        return source_codes, target_codes, values, link_colors

    @staticmethod
    def _get_node_positions(
        node_labels: list, data_frame: pd.DataFrame, n_layers: int, other_id: str
    ):
        """
        For each layer, compute x,y positions so that 'Other' lumps get placed 
        at the same y, and real nodes get spread top-to-bottom.
        """
        # Group by node
        grouped_data = data_frame.groupby("source_w_other", as_index=False).agg(
            {"source_layer": "min", "value": "mean"}
        )

        final_positions = pd.DataFrame()
        for layer_idx in range(n_layers):
            layer_slice = grouped_data[grouped_data["source_layer"] == layer_idx].copy()
            # Separate out "other" from "real"
            other_slice = layer_slice[layer_slice["source_w_other"].str.startswith(other_id)]
            real_slice = layer_slice[~layer_slice["source_w_other"].str.startswith(other_id)]

            # Sort real nodes by value to place top->bottom
            real_slice = real_slice.sort_values("value", ascending=True)
            real_slice["rank"] = range(len(real_slice))
            if real_slice["value"].sum() != 0:
                real_slice["value"] = real_slice["value"] / real_slice["value"].sum()

            # Spread them from ~0.8 to ~0
            if len(real_slice) > 1:
                max_rank = real_slice["rank"].max()
                real_slice["y"] = 0.8 * (max_rank - real_slice["rank"]) / (max_rank + 0.001)
            else:
                # If only one node, place in middle
                real_slice["y"] = 0.4

            real_slice["x"] = (0.01 + layer_idx) / (n_layers + 1)

            # Collapse "Other" nodes into a single row at y=0.9
            if not other_slice.empty:
                other_sum = other_slice["value"].sum()
                row = pd.DataFrame(
                    [
                        [
                            f"{other_id}_l{layer_idx-1}",
                            layer_idx,
                            other_sum,
                            999,
                            0.9,
                            (0.01 + layer_idx) / (n_layers + 1),
                        ]
                    ],
                    columns=["source_w_other", "source_layer", "value", "rank", "y", "x"],
                )
                real_slice = pd.concat([real_slice, row], ignore_index=True)

            final_positions = pd.concat([final_positions, real_slice], ignore_index=True)

        # Build x,y in the same order as node_labels
        x_positions, y_positions = [], []
        for label in node_labels:
            row_match = final_positions[final_positions["source_w_other"] == label]
            if "output_node" in label:
                # Hard-code output node position
                x_positions.append(0.85)
                y_positions.append(0.5)
            else:
                # Otherwise, read from the final_positions
                if not row_match.empty:
                    x_positions.append(row_match["x"].values[0])
                    y_positions.append(row_match["y"].values[0])
                else:
                    # fallback if something is missing
                    x_positions.append(0.5)
                    y_positions.append(0.5)

        return x_positions, y_positions
