
from sklearn import preprocessing
import pandas as pd
import numpy as np


def fit_protein_matrix_to_network_input(QM: pd.DataFrame, features) -> pd.DataFrame:
    nr_proteins_in_matrix = len(QM.index)
    if len(features) > nr_proteins_in_matrix:
        RN_df = pd.DataFrame(features, columns=['Protein'])
        QM = QM.merge(RN_df, how='right', on='Protein')
    if len(features) > 0:
        QM.set_index('Protein', inplace=True)
        QM = QM.loc[features]
    return QM


def generate_data(QM: pd.DataFrame, design_matrix: pd.DataFrame):
    GroupOneCols = design_matrix[design_matrix['group']
                                 == 1]['sample'].values
    GroupTwoCols = design_matrix[design_matrix['group']
                                 == 2]['sample'].values

    df1 = QM[GroupOneCols].T
    df2 = QM[GroupTwoCols].T
    y = np.array([0 for x in GroupOneCols] + [1 for x in GroupTwoCols])
    X = pd.concat([df1, df2]).fillna(0).to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)
    return X, y
