from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch


class BINNDataLoader:
    """
    A utility class for aligning data to the BINN network, preparing train/validation splits, 
    and creating PyTorch DataLoaders.
    """

    def __init__(self):
        pass

    def align_to_network(
        self,
        data_matrix: pd.DataFrame,
        binn_network,
        feature_column: str = "Protein",
    ) -> pd.DataFrame:
        """
        Align the input data matrix to the features expected by the BINN network.

        Args:
            data_matrix (pd.DataFrame): Raw data matrix.
            binn_network: The BINN model instance (for extracting `inputs`).
            feature_column (str): Column name for feature identifiers in 'data_matrix'.

        Returns:
            pd.DataFrame: Aligned data matrix with rows matching the BINN's expected features, 
                          filling missing features with zeros if needed.
        """
        features = binn_network.inputs  # Features expected by BINN
        dm = data_matrix.copy()

        # Ensure all expected features are present
        if len(features) > len(dm.index):
            missing_features = set(features) - set(dm[feature_column])
            features_df = pd.DataFrame(missing_features, columns=[feature_column])
            dm = pd.concat([dm, features_df], axis=0, ignore_index=True)

        # Reindex to ensure the row order matches BINN's feature order
        dm.set_index(feature_column, inplace=True)
        dm = dm.reindex(features).fillna(0)
        return dm

    def prepare_training_data(
        self,
        aligned_data: pd.DataFrame,
        design_matrix: pd.DataFrame,
        group_column: str = "group",
        sample_column: str = "sample",
        validation_split: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """
        Prepare training and validation data (X, y).

        Args:
            aligned_data (pd.DataFrame): Data aligned to BINN's feature order.
            design_matrix (pd.DataFrame): Contains group and sample identifiers.
            group_column (str): Column indicating class/group labels.
            sample_column (str): Column indicating sample identifiers.
            validation_split (float): Fraction of data to use for validation. Set to 0 for no validation split.
            random_state (int): Random seed for reproducibility.

        Returns:
            dict: Contains 'train': (X_train, y_train) and optionally 'val': (X_val, y_val).
        """
        # Map group labels to integers
        groups = design_matrix[group_column].unique()
        group_to_label = {g: i for i, g in enumerate(sorted(groups))}
        print(f"Mapping group labels: {group_to_label}")

        # Prepare X and y
        group_samples = {
            group: design_matrix[design_matrix[group_column] == group][sample_column].values.tolist()
            for group in groups
        }
        X_list = [aligned_data[samples].T for samples in group_samples.values()]
        X = pd.concat(X_list).fillna(0).to_numpy()
        y = np.concatenate(
            [[group_to_label[group]] * len(samples) for group, samples in group_samples.items()]
        )

        # Standardize features
        X = preprocessing.StandardScaler().fit_transform(X)

        # Split into train and validation sets
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=random_state, stratify=y
            )
            return {"train": (X_train, y_train), "val": (X_val, y_val)}
        else:
            return {"train": (X, y)}

    def create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from feature matrix (X) and labels (y).

        Args:
            X (np.ndarray): Feature matrix (num_samples x num_features).
            y (np.ndarray): Target labels (num_samples,).
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle data during loading.

        Returns:
            DataLoader: A PyTorch DataLoader.
        """
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
