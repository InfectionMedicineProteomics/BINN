from sklearn import preprocessing
from binn import BINN, Network, BINNExplainer
import numpy as np
import torch
from binn import BINN, Network
import pandas as pd
import pytorch_lightning


class RecursivePathwayElimination():

    def __init__(self, model: BINN, explainer: BINNExplainer):
        self.model = model
        self.network = model.network
        self.pathways = self.network.pathways
        self.mapping = self.network.unaltered_mapping
        self.n_layers = model.n_layers
        self.explainer = explainer
        self.input_data = self.network.input_data

    def fit(self, protein_matrix, design_matrix, dataloader, nr_iterations: int = 20, max_epochs: int = 50,  clip_threshold=1e-5, learning_rate=0.01):
        models = []
        model = self.model
        fitted_protein_matrix = self._fit_data_matrix_to_network_input(
            protein_matrix.reset_index(), features=self.network.inputs)
        X, y = self._generate_data(fitted_protein_matrix,
                                   design_matrix=design_matrix)
        dataloader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y)),
                                                 batch_size=8,
                                                 num_workers=12,
                                                 shuffle=True)
        for _ in range(nr_iterations):
            early_stopping = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss", mode="min")
            trainer = pytorch_lightning.Trainer(
                max_epochs=max_epochs, log_every_n_steps=10, callbacks=[early_stopping])
            trainer.fit(model, dataloader)
            explainer = BINNExplainer(model)
            test_data = torch.Tensor(X)
            background_data = torch.Tensor(X)
            for wanted_layer in range(self.n_layers):
                shap_dict = explainer._explain_layer(
                    test_data, background_data, wanted_layer)
                values = shap_dict['shap_values']
                values = np.abs(np.array(values))
                values = np.mean(values, axis=1)
                values = np.sum(values, axis=0)
                shap_list = [(feature, value) for feature,
                             value in zip(shap_dict['features'], values)]
                shap_list = sorted(shap_list, key=lambda x: x[1])
                features_that_are_zero = [x[0]
                                          for x in shap_list if x[1] < clip_threshold]
                n_features_that_are_zero = len(features_that_are_zero)
                shap_list = shap_list[n_features_that_are_zero:]
                total_n_elements = len(shap_list)
                n_features_to_remove = int(total_n_elements*0.05)
                features_to_remove = [x[0]
                                      for x in shap_list[:n_features_to_remove]]

                self.pathways = self.pathways[~self.pathways['parent'].isin(
                    features_to_remove)]
                self.pathways = self.pathways[~self.pathways['child'].isin(
                    features_to_remove)]
                print(len(self.pathways))

            self.network = Network(input_data=self.input_data,
                                   pathways=self.pathways, mapping=self.mapping)
            models.append(model)
            model = BINN(
                network=self.network,
                n_layers=4,
                dropout=0.2,
                validate=True,
                residual=True,
                scheduler='plateau',
                learning_rate=learning_rate
            )
            fitted_protein_matrix = self._fit_data_matrix_to_network_input(
                protein_matrix.reset_index(), features=self.network.inputs)
            X, y = self._generate_data(fitted_protein_matrix,
                                       design_matrix=design_matrix)
            dataloader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.Tensor(X), torch.LongTensor(y)),
                                                     batch_size=8,
                                                     num_workers=12,
                                                     shuffle=True)

    def _fit_data_matrix_to_network_input(self, data_matrix: pd.DataFrame, features, feature_column="Protein") -> pd.DataFrame:
        nr_features_in_matrix = len(data_matrix.index)
        if len(features) > nr_features_in_matrix:
            features_df = pd.DataFrame(features, columns=[feature_column])
            data_matrix = data_matrix.merge(
                features_df, how='right', on=feature_column)
        if len(features) > 0:
            data_matrix.set_index(feature_column, inplace=True)
            data_matrix = data_matrix.loc[features]
        return data_matrix

    def _generate_data(self, data_matrix: pd.DataFrame, design_matrix: pd.DataFrame, groups=[1, 2]):
        y = []
        dfs = []
        for i, group in enumerate(groups):
            group_columns = design_matrix[design_matrix['group']
                                          == group]['sample'].values
            df = data_matrix[group_columns].T
            dfs.append(df)
            y += [i for _ in group_columns]
        y = np.array(y)
        X = pd.concat(dfs).fillna(0).to_numpy()
        X = preprocessing.StandardScaler().fit_transform(X)
        return X, y
