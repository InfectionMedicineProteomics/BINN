import matplotlib.pyplot as plt
import shap
import numpy as np
import torch
from binn import BINN
import pandas as pd


class BINNExplainer:
    def __init__(self, model: BINN):

        self.model = model

    def explain(self, test_data: torch.Tensor, background_data: torch.Tensor):

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
            # first dim: positive vs negative class, second dim: for each test data, third dim: for each feature
            sv = np.asarray(sv)
            sv = abs(sv)
            # mean(|shap_value|) = impact on model class
            sv_mean = np.mean(sv, axis=1)
            for f in range(sv_mean.shape[-1]):
                connections = cm[cm.index == features[f]]
                connections = connections.loc[
                    :, (connections != 0).any(axis=0)
                ]  # get targets and append to target
                for target in connections:
                    feature_dict["source"].append(
                        f"{features[f]}_{curr_layer}")
                    feature_dict["target"].append(f"{target}_{curr_layer + 1}")
                    feature_dict["value"].append(sv_mean[0][f])
                    feature_dict["type"].append(0)
                    feature_dict["source"].append(
                        f"{features[f]}_{curr_layer}")
                    feature_dict["target"].append(f"{target}_{curr_layer + 1}")
                    feature_dict["value"].append(sv_mean[1][f])
                    feature_dict["type"].append(1)
                    feature_dict["source layer"].append(curr_layer)
                    feature_dict["source layer"].append(curr_layer)
                    feature_dict["target layer"].append(curr_layer + 1)
                    feature_dict["target layer"].append(curr_layer + 1)
            curr_layer += 1
        df = pd.DataFrame(data=feature_dict)
        return df

    def explain_input(
        self, test_data: torch.Tensor, background_data: torch.Tensor, layer: int
    ):

        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(test_data)

        shap_dict = {
            "features": self.model.column_names[0], "shap_values": shap_values}

        return shap_dict

    def _explain_layers(
        self, background_data: torch.Tensor, test_data: torch.Tensor
    ) -> dict:

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
