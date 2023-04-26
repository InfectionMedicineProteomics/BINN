import shap
import numpy as np
import torch
from binn import BINN
import pandas as pd
import pytorch_lightning


class BINNExplainer:
    """
    A class for explaining the predictions of a BINN model using SHAP values.

    Args:
        model (BINN): A trained BINN model.
    """

    def __init__(self, model: BINN):

        self.model = model

    def explain(self, test_data: torch.Tensor, background_data: torch.Tensor):
        """
        Generates SHAP explanations for a given test_data by computing the Shapley values for each feature using
        the provided background_data. The feature importances are then aggregated and returned in a pandas dataframe.

        Args:
            test_data (torch.Tensor): The input data for which to generate the explanations.
            background_data (torch.Tensor): The background data to use for computing the Shapley values.

        Returns:
            pd.DataFrame: A dataframe containing the aggregated SHAP feature importances.
        """
        shap_dict = self._explain_layers(background_data, test_data)
        feature_dict = {
            "source": [],
            "target": [],
            "value": [],
            "type": [],
            "source layer": [],
            "target layer": [],
        }
        connectivity_matrices = self.model.get_connectivity_matrices()
        curr_layer = 0
        for sv, features, cm in zip(
            shap_dict["shap_values"], shap_dict["features"], connectivity_matrices
        ):
            sv = np.asarray(sv)
            sv = abs(sv)
            sv_mean = np.mean(sv, axis=1)

            for f in range(sv_mean.shape[-1]):
                n_classes = sv_mean.shape[0]
                connections = cm[cm.index == features[f]]
                connections = connections.loc[
                    :, (connections != 0).any(axis=0)
                ]  # get targets and append to target
                for target in connections:
                    for curr_class in range(n_classes):
                        feature_dict["source"].append(
                            f"{features[f]}_{curr_layer}")
                        feature_dict["target"].append(
                            f"{target}_{curr_layer + 1}")
                        feature_dict["value"].append(sv_mean[curr_class][f])
                        feature_dict["type"].append(curr_class)
                        feature_dict["source layer"].append(curr_layer)
                        feature_dict["target layer"].append(curr_layer + 1)
            curr_layer += 1
        df = pd.DataFrame(data=feature_dict)
        return df

    def explain_average(self,
                        test_data: torch.Tensor,
                        background_data: torch.Tensor,
                        nr_iterations: int,
                        max_epochs: int,
                        dataloader) -> pd.DataFrame:
        """
        Computes the SHAP explanations for the given test_data by averaging the Shapley values over multiple iterations.
        For each iteration, the model's parameters are randomly initialized and trained on the provided data using
        the provided trainer and dataloader. The feature importances are then aggregated and returned in a pandas dataframe.

        Args:
            test_data (torch.Tensor): The input data for which to generate the explanations.
            background_data (torch.Tensor): The background data to use for computing the Shapley values.
            nr_iterations (int): The number of iterations to use for averaging the Shapley values.
            trainer: The PyTorch Lightning trainer to use for training the model.
            dataloader: The PyTorch DataLoader to use for loading the data.

        Returns:
            pd.DataFrame: A dataframe containing the aggregated SHAP feature importances.
        """
        dfs = {}
        for iteration in range(nr_iterations):
            trainer = pytorch_lightning.Trainer(max_epochs=max_epochs)
            self.model.reset_params()
            self.model.init_weights()
            trainer.fit(self.model, dataloader)
            df = self.explain(test_data, background_data)
            dfs[iteration] = df

        col_names = [f'value_{n}' for n in range(len(list(dfs.keys())))]
        values = [df.value.values for df in dfs.values()]
        values = np.array(values)
        values_mean = np.mean(values, axis=0)
        values_std = np.std(values, axis=0)
        df = dfs[0].copy()
        df.drop(columns=['value'], inplace=True)
        df[col_names] = values.T
        df['value_mean'] = values_mean
        df['values_std'] = values_std
        df['value'] = values_mean
        return df

    def explain_input(
        self, test_data: torch.Tensor, background_data: torch.Tensor, layer: int
    ) -> dict:
        """
        Computes the SHAP explanations for the given test_data for a specific layer in the model by computing the
        Shapley values for each feature using the provided background_data. The feature importances are then returned
        in a dictionary.

        Args:
            test_data (torch.Tensor): The input data for which to generate the explanations.
            background_data (torch.Tensor): The background data to use for computing the Shapley values.
            layer (int): The index of the layer for which to compute the SHAP explanations.

        Returns:
            dict: A dictionary containing the SHAP feature importances.
        """

        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(test_data)

        shap_dict = {
            "features": self.model.column_names[0], "shap_values": shap_values}

        return shap_dict

    def _explain_layers(
        self, background_data: torch.Tensor, test_data: torch.Tensor
    ) -> dict:
        """
        Helper method to compute SHAP explanations for each layer in the model.

        Args:
            background_data (torch.Tensor): The background data to use for computing the Shapley values.
            test_data (torch.Tensor): The input data for which to generate the explanations.

        Returns:
            dict: A dictionary containing the SHAP feature importances for each layer.
        """
        feature_index = 0

        intermediate_data = test_data

        shap_dict = {"features": [], "shap_values": []}

        for name, layer in self.model.layers.named_children():

            if isinstance(layer, torch.nn.Linear) and (
                not "Residual" in name or "final" in name
            ):
                explainer = shap.DeepExplainer(
                    (self.model, layer), background_data)
                shap_values = explainer.shap_values(test_data)
                shap_dict["features"].append(
                    self.model.layer_names[feature_index])
                shap_dict["shap_values"].append(shap_values)
                feature_index += 1

                intermediate_data = layer(intermediate_data)
            if (
                isinstance(layer, torch.nn.Tanh)
                or isinstance(layer, torch.nn.ReLU)
                or isinstance(layer, torch.nn.LeakyReLU)
            ):
                intermediate_data = layer(intermediate_data)
        return shap_dict
