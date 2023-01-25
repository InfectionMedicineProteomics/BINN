import matplotlib.pyplot as plt
import shap
import numpy as np
import torch
from NN import BINN
import pandas as pd


def shap_for_layers(model: BINN, background_data: torch.Tensor, test_data: torch.Tensor, plot=True) -> dict:
    feature_index = 0
    intermediate_data = test_data
    shap_dict = {'features': [], 'shap_values': []}
    for name, layer in model.layers.named_children():
        if isinstance(layer, torch.nn.Linear) and (not "Residual" in name or "final" in name):
            feature_names = model.column_names[feature_index]
            explainer = shap.DeepExplainer((model, layer), background_data)
            shap_values = explainer.shap_values(test_data)
            shap_dict['features'].append(model.column_names[feature_index])
            shap_dict['shap_values'].append(shap_values)
            if plot:
                shap.summary_plot(shap_values, intermediate_data,
                                  feature_names=feature_names, max_display=30, plot_size=[15, 6])
                plt.savefig(
                    f'shap_summary_{feature_index}.jpg', dpi=200)
            feature_index += 1
            plt.clf()
            intermediate_data = layer(intermediate_data)
        if isinstance(layer, torch.nn.Tanh) or isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.LeakyReLU):
            intermediate_data = layer(intermediate_data)
    return shap_dict


def ExplainBINN(model: BINN, test_data: torch.Tensor, background_data: torch.Tensor) -> pd.DataFrame:
    shap_dict = shap_for_layers(model, background_data, test_data)
    feature_dict = {'source': [], 'target': [], 'value': [],
                    'type': [], 'source layer': [], 'target layer': []}
    connectivity_matrices = model.get_connectivity_matrices()
    curr_layer = 0
    for sv, features, cm in zip(shap_dict['shap_values'], shap_dict['features'], connectivity_matrices):
        # first dim: positive vs negative class, second dim: for each test data, third dim: for each feature
        sv = np.asarray(sv)
        sv = abs(sv)
        # mean(|shap_value|) = impact on model class
        sv_mean = np.mean(sv, axis=1)
        for f in range(sv_mean.shape[-1]):
            connections = cm[cm.index == features[f]]
            connections = connections.loc[:, (connections != 0).any(
                axis=0)]  # get targets and append to target
            for target in connections:
                feature_dict['source'].append(
                    f"{features[f]}_{curr_layer}")
                feature_dict['target'].append(f"{target}_{curr_layer+1}")
                feature_dict['value'].append(sv_mean[0][f])
                feature_dict['type'].append(0)
                feature_dict['source'].append(
                    f"{features[f]}_{curr_layer}")
                feature_dict['target'].append(f"{target}_{curr_layer+1}")
                feature_dict['value'].append(sv_mean[1][f])
                feature_dict['type'].append(1)
                feature_dict['source layer'].append(curr_layer)
                feature_dict['source layer'].append(curr_layer)
                feature_dict['target layer'].append(curr_layer+1)
                feature_dict['target layer'].append(curr_layer+1)
        curr_layer += 1
    df = pd.DataFrame(data=feature_dict)
    return df
