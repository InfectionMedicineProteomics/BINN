import shap
import numpy as np
import torch
import pandas as pd
from torch import nn
import networkx as nx
from typing import Dict, Tuple
from binn import BINN, BINNTrainer


class BINNExplainer:
    """
    A class for explaining the predictions of a BINN model using SHAP values,
    assuming we can gather all samples from a dictionary of DataLoaders.

    Usage:
    ------
    1. Initialize with a trained BINN model.
    2. Call `explain(dataloaders, split="train")` or
       `explain(dataloaders, split=None)` to produce SHAP-based explanations
       for either one or all splits.
    3. If you wish to do multiple re-initializations and training, call `explain`
       with `nr_iterations` and a trainer (pure PyTorch).
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : nn.Module
            A trained BINN model.
        """
        self.model = model

    def update_model(self, model: BINN):
        """Update the current BINN model for explanations."""
        self.model = model

    def explain_single(self, dataloaders: dict, split: str = None, normalization_method: str = "subgraph") -> pd.DataFrame:
        """
        Gathers all samples from the specified DataLoader(s),
        uses them for both background and test data in SHAP,
        and returns a DataFrame of explanations.

        Parameters
        ----------
        dataloaders : dict
            A dictionary containing one or more DataLoaders, e.g. {"train": train_dl, "val": val_dl}.
        split : str, optional
            If provided, gather samples only from dataloaders[split]. If None, gather from all.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns:
              ['source_layer', 'target_layer', 'source_node', 'target_node',
               'class_idx', 'importance']
            capturing the mean absolute SHAP importance for each connection in the BINN.
        """
        all_inputs, _ = self._gather_all_from_dataloader(dataloaders, split=split)
        all_inputs = all_inputs.to(self.model.device)

        shap_dict = self._explain_layers(all_inputs, all_inputs)
        explanation_df = self._shap_to_dataframe(shap_dict)
        if normalization_method:
            explanation_df = self.normalize_importances(explanation_df, method=normalization_method)
        return explanation_df

    def explain(
        self,
        dataloaders: dict,
        nr_iterations: int,
        num_epochs: int,
        trainer: BINNTrainer,
        split: str = None,
        normalization_method: str = "subgraph"
    ) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
        """
        Re-initializes the BINN model multiple times, trains it using the given trainer,
        computes SHAP for each iteration, then aggregates the results.

        Parameters
        ----------
        dataloaders : dict
            Dictionary of DataLoaders (e.g. {"train": train_dl, "val": val_dl}).
        nr_iterations : int
            Number of random re-initializations/training runs to average over.
        num_epochs : int
            How many epochs to train each iteration.
        trainer :
            A trainer that runs a pure PyTorch loop (trainer.fit(dataloaders, num_epochs)).
        split : str, optional
            The specific split to gather data from for SHAP. If None, use all splits.

        Returns
        -------
        (pd.DataFrame, dict)
            - A DataFrame with columns for each iteration’s 'importance' plus
              'importance_mean'/'importance_std'.
            - A dict containing training metrics from each iteration (e.g., accuracy/loss).
        """
        all_dfs = {}

        for iteration in range(nr_iterations):
            print(f"[BINNExplainer] Iteration {iteration+1}/{nr_iterations}...")

            # Re-init model params
            self.model.apply(_reset_params)
            self.model.apply(_init_weights)

            # Use the given trainer to train
            trainer.update_model(self.model)
            trainer.fit(dataloaders, num_epochs)

            # Then compute explanations with the newly trained model
            iteration_df = self.explain_single(dataloaders, split=split)
            all_dfs[iteration] = iteration_df

        combined_df = self._combine_iterations(all_dfs)
        if normalization_method:
            combined_df = self.normalize_importances(combined_df, method=normalization_method)
        return combined_df

    def normalize_importances(
        self,
        explanation_df: pd.DataFrame,
        method: str = "subgraph",
    ) -> pd.DataFrame:
        """
        Normalizes the 'importance' (or 'value') column in the DataFrame
        using either 'fan' or 'subgraph' logic:

        - fan:    importance / log2( fan_in + fan_out + 1 )
        - subgraph: importance / log2( upstream_subgraph_nodes + downstream_subgraph_nodes )

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least [source_node, target_node, value_col].
        method : {"fan", "subgraph"}
            The normalization strategy.

        Returns
        -------
        pd.DataFrame
            A **copy** of the input DataFrame with a newly normalized
            column, `'normalized_value'`.
        """
        explanation_df = explanation_df.copy()

        if "mean_importance" in explanation_df.columns:
            value_col = "mean_importance"
        else:
            value_col = "importance"

        G = nx.DiGraph()
        for _, row in explanation_df.iterrows():
            src = row["source_node"]
            tgt = row["target_node"]
            if not G.has_node(src):
                G.add_node(src)
            if not G.has_node(tgt):
                G.add_node(tgt)
            G.add_edge(src, tgt)

        if method == "fan":
            fan_in = {n: 0 for n in G.nodes()}
            fan_out = {n: 0 for n in G.nodes()}
            for n in G.nodes():
                fan_in[n] = G.in_degree(n)
                fan_out[n] = G.out_degree(n)

        elif method == "subgraph":
            G_reverse = G.reverse(copy=True)

            upstream_count = {}
            downstream_count = {}

            for node in G.nodes():
                down_nodes = nx.descendants(G, node)
                downstream_count[node] = len(down_nodes)
                up_nodes = nx.descendants(G_reverse, node)
                upstream_count[node] = len(up_nodes)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


        norm_vals = []
        for _, row in explanation_df.iterrows():
            node = row["source_node"] 
            raw_imp = row[value_col]

            if method == "fan":
                fi = fan_in[node]
                fo = fan_out[node]
                total = fi + fo + 1
                new_val = raw_imp / (np.log2(total) if total > 1 else 1.0)

            else:  # subgraph
                ups = upstream_count[node]
                downs = downstream_count[node]
                total = ups + downs
                new_val = raw_imp / (np.log2(total) if total > 1 else 1.0)

            norm_vals.append(new_val)

        explanation_df["normalized_importance"] = norm_vals
        return explanation_df

    def _explain_layers(
        self, background_data: torch.Tensor, test_data: torch.Tensor
    ) -> Dict[str, list]:
        """
        Computes SHAP explanations for each 'Linear' layer in the BINN.

        Returns
        -------
        dict
            {
                "features": List[List[str]]  # per-layer feature names
                "shap_values": List[np.ndarray]  # per-layer SHAP arrays
            }
        """
        shap_results = {"features": [], "shap_values": []}
        layer_idx = 0

        for name, layer in self.model.layers.named_children():
            if isinstance(layer, nn.Linear):
                explainer = shap.DeepExplainer((self.model, layer), background_data)
                svals = explainer.shap_values(test_data)
                shap_results["features"].append(self.model.layer_names[layer_idx])
                shap_results["shap_values"].append(svals)
                layer_idx += 1

        return shap_results

    def _shap_to_dataframe(self, shap_dict: Dict[str, list]) -> pd.DataFrame:
        """
        Convert raw shap_dict from `_explain_layers` to a tidy DataFrame
        containing connection-level SHAP importance.

        Columns: ['source_layer', 'target_layer', 'source_node', 'target_node',
                'class_idx', 'importance']
        """
        connectivity_mats = self.model.connectivity_matrices

        all_rows = []
        current_layer = 0

        for svals, feats, cm in zip(
            shap_dict["shap_values"], shap_dict["features"], connectivity_mats
        ):
            svals = np.asarray(svals)
            svals = np.abs(svals)
            svals_mean = svals.mean(
                axis=0
            )  # average over samples => (num_classes, num_feats)

            num_feats, num_classes = svals_mean.shape

            for feat_idx, feat_name in enumerate(feats):
                if feat_name not in cm.index:
                    continue
                row_conn = cm.loc[[feat_name], :]
                # drop columns with all zero edges (these are pruned edges)
                row_conn = row_conn.loc[:, (row_conn != 0).any(axis=0)]
                if row_conn.empty:
                    continue

                for target_name in row_conn.columns:
                    # for each class
                    for class_idx in range(num_classes):
                        importance_val = float(svals_mean[feat_idx, class_idx])
                        all_rows.append(
                            {
                                "source_layer": current_layer,
                                "target_layer": current_layer + 1,
                                "source_node": feat_name,
                                "target_node": target_name,
                                "class_idx": class_idx,
                                "importance": importance_val,
                            }
                        )
            current_layer += 1

        df = pd.DataFrame(all_rows)
        return df


    def _combine_iterations(
        self, results_dict: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine multiple iteration results into a single DataFrame,
        computing mean and std of 'importance'.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with new columns: importance_<iter>,
            importance_mean, importance_std, plus the final 'importance' set to the mean.
        """
        first_key = list(results_dict.keys())[0]
        merged_df = results_dict[first_key].copy()

        iteration_cols = []
        for iteration_idx, df_iter in results_dict.items():
            col_name = f"importance_{iteration_idx}"
            merged_df[col_name] = df_iter["importance"].values
            iteration_cols.append(col_name)

        arr_vals = merged_df[iteration_cols].values  # shape: (n_rows, n_iters)
        mean_vals = arr_vals.mean(axis=1)
        std_vals = arr_vals.std(axis=1)

        merged_df["importance_mean"] = mean_vals
        merged_df["importance_std"] = std_vals
        merged_df["importance"] = mean_vals
        return merged_df

    def _gather_all_from_dataloader(self, dataloaders: dict, split: str = None):
        """
        Gather all inputs from a specified split or from all splits if split is None.

        Args:
            dataloaders (dict): e.g. {"train": train_dl, "val": val_dl}
            split (str): if provided, use only dataloaders[split].
                         if None, gather from all splits.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            (all_inputs, all_targets) concatenated across all (or one) splits.
        """
        all_x = []
        all_y = []

        if split is not None:
            # Check if the split exists
            if split not in dataloaders:
                raise ValueError(
                    f"Split '{split}' not found in dataloaders. "
                    f"Available splits: {list(dataloaders.keys())}"
                )
            dl = dataloaders[split]
            for batch in dl:
                inputs, targets = batch[0], batch[1]
                all_x.append(inputs)
                all_y.append(targets)
        else:
            # Concatenate all splits
            for name, dl in dataloaders.items():
                for batch in dl:
                    inputs, targets = batch[0], batch[1]
                    all_x.append(inputs)
                    all_y.append(targets)

        if not all_x:
            raise ValueError("No samples found in the specified DataLoaders.")

        X = torch.cat(all_x, dim=0)
        Y = torch.cat(all_y, dim=0)
        return X, Y


#
# Helper functions for re-init
#
def _reset_params(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
