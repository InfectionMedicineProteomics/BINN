import shap
import numpy as np
import torch
from binn import BINN
import pandas as pd
import lightning.pytorch as pl
from .feature_selection import RecursivePathwayElimination


class BINNExplainer:
    """
    A class for explaining the predictions of a BINN model using SHAP values.

    Args:
        model (BINN): A trained BINN model.
    """

    def __init__(self, model: BINN):
        self.model = model

    def update_model(self, model: BINN):
        self.model = model

    def explain(self, test_data: torch.tensor, background_data: torch.tensor):
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
            "source name": [],
            "target name": [],
            "value": [],
            "type": [],
            "source layer": [],
            "target layer": [],
        }
        connectivity_matrices = self.model.get_connectivity_matrices()
        feature_id_mapping = {}

        feature_id = 0
        for layer, layer_features in enumerate(shap_dict["features"]):
            for feature in layer_features:
                feature_id += 1
                feature_id_mapping[f"{feature}_{layer}"] = feature_id
        
        feature_id_mapping[f"root_{layer+1}"] = 0

        curr_layer = 0
        for sv, features, cm in zip(
            shap_dict["shap_values"], shap_dict["features"], connectivity_matrices
        ):
            sv = np.asarray(sv)
            sv = abs(sv)
            sv_mean = np.mean(sv, axis=1)

            for feature in range(sv_mean.shape[-1]):
                n_classes = sv_mean.shape[0]
                connections = cm[cm.index == features[feature]]
                connections = connections.loc[
                    :, (connections != 0).any(axis=0)
                ]  # get targets and append to target
                for target in connections:
                    for curr_class in range(n_classes):
                        feature_dict["source"].append(
                            feature_id_mapping[f"{features[feature]}_{curr_layer}"]
                        )
                        feature_dict["target"].append(feature_id_mapping[f"{target}_{curr_layer+1}"])
                        feature_dict["source name"].append(features[feature])
                        feature_dict["target name"].append(target)
                        feature_dict["value"].append(sv_mean[curr_class][feature])
                        feature_dict["type"].append(curr_class)
                        feature_dict["source layer"].append(curr_layer)
                        feature_dict["target layer"].append(curr_layer + 1)
            curr_layer += 1
        df = pd.DataFrame(data=feature_dict)
        return df

    def fast_train(self, dataloader, num_epochs, optimizer):
        return_dict = {"accuracies":[], "losses":[], "epoch":[]}
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_accuracy = 0

            for _, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device).type(torch.LongTensor)
                optimizer.zero_grad()
                outputs = self.model(inputs).to(self.model.device)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_accuracy += torch.sum(
                    torch.argmax(outputs, axis=1) == targets
                ) / len(targets)

            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)
            return_dict["accuracies"].append(avg_accuracy.numpy().tolist())
            return_dict["losses"].append(avg_loss)
            return_dict["epoch"].append(epoch)
        print(
            f"Final epoch: Average Accuracy {avg_accuracy:.2f}, Average Loss: {avg_loss:.2f}"
        )
        return self.model, return_dict

    def explain_average(
        self,
        test_data: torch.Tensor,
        background_data: torch.Tensor,
        nr_iterations: int,
        max_epochs: int,
        dataloader,
        fast_train: bool,
    ) -> (pd.DataFrame, dict):
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
            (pd.DataFrame, dict): A dataframe containing the aggregated SHAP feature importances and a dict containing training metrics.
        """
        dfs = {}
        metrics_dict = {}
        for iteration in range(nr_iterations):
            print(f"Iteration {iteration}")
            self.model.reset_params()
            self.model.init_weights()
            if fast_train:
                optimizer = self.model.configure_optimizers()[0][0]
                self.model, return_dict = self.fast_train(dataloader, max_epochs, optimizer)
                metrics_dict[iteration] = return_dict
            else:
                trainer = pl.Trainer(max_epochs=max_epochs)
                trainer.fit(self.model, dataloader)
            df = self.explain(test_data, background_data)
            dfs[iteration] = df

        col_names = [f"value_{n}" for n in range(len(list(dfs.keys())))]
        values = [df.value.values for df in dfs.values()]
        values = np.array(values)
        values_mean = np.mean(values, axis=0)
        values_std = np.std(values, axis=0)
        df = dfs[0].copy()
        df.drop(columns=["value"], inplace=True)
        df[col_names] = values.T
        df["value_mean"] = values_mean
        df["values_std"] = values_std
        df["value"] = values_mean
        return df, metrics_dict

    def recursive_pathway_elimination(
        self,
        input_data,
        design_matrix,
        nr_iterations: int = 20,
        max_epochs: int = 50,
        clip_threshold=1e-5,
        constant_removal_rate=0.05,
        min_features_per_layer=3,
        early_stopping=True,
    ):
        rpe = RecursivePathwayElimination(self.model, self)
        return_dict = rpe.fit(
            input_data=input_data,
            design_matrix=design_matrix,
            nr_iterations=nr_iterations,
            max_epochs=max_epochs,
            clip_threshold=clip_threshold,
            constant_removal_rate=constant_removal_rate,
            min_features_per_layer=min_features_per_layer,
            early_stopping=early_stopping,
        )

        self.rpe_model = rpe.get_final_model()
        self.rpe_data = rpe.get_final_data()

        return return_dict

    def explain_input(
        self, test_data: torch.Tensor, background_data: torch.Tensor
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

        shap_dict = {"features": self.model.layer_names[0], "shap_values": shap_values}

        return shap_dict

    def _explain_layers(
        self, background_data: torch.tensor, test_data: torch.tensor
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
                "Residual" not in name or "final" in name
            ):
                explainer = shap.DeepExplainer((self.model, layer), background_data)
                shap_values = explainer.shap_values(test_data)
                shap_dict["features"].append(self.model.layer_names[feature_index])
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

    def _explain_layer(
        self, background_data: torch.tensor, test_data: torch.tensor, wanted_layer: int
    ) -> dict:
        intermediate_data = test_data

        shap_dict = {"features": [], "shap_values": []}
        layer_index = 0
        for name, layer in self.model.layers.named_children():
            if isinstance(layer, torch.nn.Linear) and (
                "Residual" not in name or "final" in name
            ):
                if layer_index == wanted_layer:
                    explainer = shap.DeepExplainer((self.model, layer), background_data)
                    shap_values = explainer.shap_values(test_data)
                    shap_dict["features"] += self.model.layer_names[wanted_layer]
                    shap_dict["shap_values"] += shap_values
                    return shap_dict
                layer_index += 1
                intermediate_data = layer(intermediate_data)
            if (
                isinstance(layer, torch.nn.Tanh)
                or isinstance(layer, torch.nn.ReLU)
                or isinstance(layer, torch.nn.LeakyReLU)
            ):
                intermediate_data = layer(intermediate_data)
        return shap_dict
