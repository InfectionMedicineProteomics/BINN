from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from binn import BINN, Network, BINNExplainer
import numpy as np
import torch
import pandas as pd
import pytorch_lightning


class RecursivePathwayElimination:
    def __init__(self, model: BINN, explainer: BINNExplainer):
        self.model = model
        self.n_layers = model.n_layers
        self.learning_rate = model.learning_rate
        self.weight = model.weight
        self.network = model.network
        self.pathways = self.network.pathways
        self.mapping = self.network.unaltered_mapping
        self.n_layers = model.n_layers
        self.explainer = explainer
        self.input_data = self.network.input_data

    def fit(
        self,
        protein_matrix,
        design_matrix,
        nr_iterations: int = 20,
        max_epochs: int = 50,
        clip_threshold=1e-5,
        constant_removal_rate=0.05,
        min_features_per_layer=3,
    ):
        return_dict = {
            "models": [],
            "val_acc": [],
            "val_loss": [],
            "iteration": [],
            "pathways": [],
            "epochs": [],
            "trainable_params": [],
        }
        self.fitted_protein_matrix = self._fit_data_matrix_to_network_input(
            protein_matrix.reset_index(), features=self.network.inputs
        )
        X_train, X_test, y_train, y_test = self._generate_data(
            self.fitted_protein_matrix, design_matrix=design_matrix
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.Tensor(X_train), torch.LongTensor(y_train)
            ),
            batch_size=8,
            num_workers=12,
            shuffle=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.Tensor(X_test), torch.LongTensor(y_test)
            ),
            batch_size=8,
            num_workers=12,
        )
        model = self.model
        for iteration in range(nr_iterations):
            self.model = model
            total_removed_features = 0
            pathway_dict = {}
            early_stopping = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
                patience=10, min_delta=0.001, monitor="val_loss", mode="min"
            )
            trainer = pytorch_lightning.Trainer(
                max_epochs=max_epochs, log_every_n_steps=5, callbacks=[early_stopping]
            )
            trainer.fit(self.model, train_dataloader, val_dataloader)
            return_dict["epochs"].append(self.model.current_epoch)
            return_dict["trainable_params"].append(self.model.trainable_params)
            val_dict = trainer.validate(self.model, val_dataloader)
            return_dict["models"].append(self.model)
            return_dict["val_acc"].append(val_dict[0]["val_acc"])
            return_dict["val_loss"].append(val_dict[0]["val_loss"])
            return_dict["iteration"].append(iteration)
            explainer = BINNExplainer(self.model)
            self.test_data = torch.Tensor(np.concatenate([X_train, X_test], axis=0))
            self.background_data = torch.Tensor(
                np.concatenate([X_train, X_test], axis=0)
            )
            for wanted_layer in range(self.n_layers + 1):
                shap_dict = explainer._explain_layer(
                    self.test_data, self.background_data, wanted_layer
                )
                values = shap_dict["shap_values"]
                values = np.abs(np.array(values))
                values = np.mean(values, axis=1)
                values = np.sum(values, axis=0)
                shap_list = [
                    (feature, value)
                    for feature, value in zip(shap_dict["features"], values)
                ]
                if len(shap_list) <= min_features_per_layer:
                    print(f"Min features per layer ({min_features_per_layer}) reached")
                    pathway_dict[wanted_layer] = self.pathways.copy()
                else:
                    shap_list = sorted(shap_list, key=lambda x: x[1])
                    features_below_clip = [
                        x[0] for x in shap_list if x[1] < clip_threshold
                    ]
                    n_features_below_clip = len(features_below_clip)
                    shap_list = shap_list[n_features_below_clip:]
                    total_n_elements = len(shap_list)
                    n_features_to_remove = int(total_n_elements * constant_removal_rate)
                    features_to_remove = [
                        x[0] for x in shap_list[:n_features_to_remove]
                    ]
                    features_to_remove += features_below_clip
                    features_to_remove = list(set(features_to_remove))
                    self.pathways = self.pathways[
                        ~self.pathways["parent"].isin(features_to_remove)
                    ]
                    self.pathways = self.pathways[
                        ~self.pathways["child"].isin(features_to_remove)
                    ]
                    pathway_dict[wanted_layer] = self.pathways.copy()
                    print(f"Connections in layer {wanted_layer}: ", len(self.pathways))
                    print(
                        f"Removed pathways in layer {wanted_layer}: \
                        {len(features_to_remove)} \
                            out of {total_n_elements} \
                        "
                    )
                    print(features_to_remove)
                    total_removed_features += len(features_to_remove)
            return_dict["pathways"].append(pathway_dict)

            if total_removed_features == 0:
                print(f"Stopped at iteration: {iteration}")
                return return_dict

            self.network = Network(
                input_data=self.input_data, pathways=self.pathways, mapping=self.mapping
            )

            model = BINN(
                network=self.network,
                n_layers=self.n_layers,
                dropout=0.2,
                weight=self.weight,
                validate=True,
                residual=True,
                scheduler="plateau",
                learning_rate=self.learning_rate,
            )
            self.fitted_protein_matrix = self._fit_data_matrix_to_network_input(
                protein_matrix.reset_index(), features=self.network.inputs
            )
            X_train, X_test, y_train, y_test = self._generate_data(
                self.fitted_protein_matrix, design_matrix=design_matrix
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(
                    torch.Tensor(X_train), torch.LongTensor(y_train)
                ),
                batch_size=8,
                num_workers=12,
                shuffle=True,
            )
            val_dataloader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(
                    torch.Tensor(X_test), torch.LongTensor(y_test)
                ),
                batch_size=8,
                num_workers=12,
            )
        return return_dict

    def get_final_model(self):
        return self.model

    def get_protein_matrix(self):
        return self.fitted_protein_matrix

    def _fit_data_matrix_to_network_input(
        self, data_matrix: pd.DataFrame, features, feature_column="Protein"
    ) -> pd.DataFrame:
        nr_features_in_matrix = len(data_matrix.index)
        if len(features) > nr_features_in_matrix:
            features_df = pd.DataFrame(features, columns=[feature_column])
            data_matrix = data_matrix.merge(features_df, how="right", on=feature_column)
        if len(features) > 0:
            data_matrix.set_index(feature_column, inplace=True)
            data_matrix = data_matrix.loc[features]
        return data_matrix

    def _generate_data(
        self,
        data_matrix: pd.DataFrame,
        design_matrix: pd.DataFrame,
        groups=[1, 2],
        test_size=0.25,
    ):
        y = []
        dfs = []
        for i, group in enumerate(groups):
            group_columns = design_matrix[design_matrix["group"] == group][
                "sample"
            ].values
            df = data_matrix[group_columns].T
            dfs.append(df)
            y += [i for _ in group_columns]
        y = np.array(y)
        X = pd.concat(dfs).fillna(0).to_numpy()
        X = preprocessing.StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test
