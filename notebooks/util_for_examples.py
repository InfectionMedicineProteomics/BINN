
from sklearn import preprocessing
import pandas as pd
import numpy as np


def fit_data_matrix_to_network_input(data_matrix: pd.DataFrame, features, feature_column="Protein") -> pd.DataFrame:
    nr_features_in_matrix = len(data_matrix.index)
    if len(features) > nr_features_in_matrix:
        features_df = pd.DataFrame(features, columns=[feature_column])
        data_matrix = data_matrix.merge(
            features_df, how='right', on=feature_column)
    if len(features) > 0:
        data_matrix.set_index(feature_column, inplace=True)
        data_matrix = data_matrix.loc[features]
    return data_matrix


def generate_data(data_matrix: pd.DataFrame, design_matrix: pd.DataFrame):
    GroupOneCols = design_matrix[design_matrix['group']
                                 == 1]['sample'].values
    GroupTwoCols = design_matrix[design_matrix['group']
                                 == 2]['sample'].values

    df1 = data_matrix[GroupOneCols].T
    df2 = data_matrix[GroupTwoCols].T
    y = np.array([0 for _ in GroupOneCols] + [1 for _ in GroupTwoCols])
    X = pd.concat([df1, df2]).fillna(0).to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)
    return X, y
