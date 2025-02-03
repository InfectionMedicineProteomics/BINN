from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch


class BINNDataLoader:
    """
    A utility class for aligning data to the BINN network, preparing train/validation splits,
    and creating PyTorch DataLoaders with a simplified user interface.
    """

    def __init__(self, binn_network):
        """
        Args:
            binn_network: The BINN model instance for feature alignment.
        """
        self.binn_network = binn_network

    def _align_to_network(
        self,
        data_matrix: pd.DataFrame,
        feature_column: str,
    ) -> pd.DataFrame:
        """
        Internal method to align the input data matrix to the BINN network's expected features.

        Args:
            data_matrix (pd.DataFrame): Raw data matrix.
            feature_column (str): Column name for feature identifiers in 'data_matrix'.

        Returns:
            pd.DataFrame: Data matrix with rows matching the BINN's expected features,
                          filling missing features with zeros if needed.
        """
        features = self.binn_network.inputs  # Features expected by BINN
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

    def _prepare_training_data(
        self,
        aligned_data: pd.DataFrame,
        design_matrix: pd.DataFrame,
        group_column: str,
        sample_column: str,
        validation_split: float,
        random_state: int,
    ) -> dict:
        """
        Internal method to prepare (X, y) for training, with optional validation split.

        Args:
            aligned_data (pd.DataFrame): Data aligned to BINN's feature order.
            design_matrix (pd.DataFrame): Contains group and sample identifiers.
            group_column (str): Column indicating class/group labels.
            sample_column (str): Column indicating sample identifiers.
            validation_split (float): Fraction of data to reserve for validation.
            random_state (int): RNG seed for reproducibility.

        Returns:
            dict: Contains 'train': (X_train, y_train) and optionally 'val': (X_val, y_val).
        """
        # Map group labels to integers
        groups = design_matrix[group_column].unique()
        group_to_label = {g: i for i, g in enumerate(sorted(groups))}
        print(f"Mapping group labels: {group_to_label}")

        # Prepare X and y
        group_samples = {
            group: design_matrix[design_matrix[group_column] == group][
                sample_column
            ].values.tolist()
            for group in groups
        }
        X_list = [aligned_data[samples].T for samples in group_samples.values()]
        X = pd.concat(X_list).fillna(0).to_numpy()
        y = np.concatenate(
            [
                [group_to_label[group]] * len(samples)
                for group, samples in group_samples.items()
            ]
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

    def create_dataloaders(
        self,
        data_matrix: pd.DataFrame,
        design_matrix: pd.DataFrame,
        feature_column: str = "Protein",
        group_column: str = "group",
        sample_column: str = "sample",
        batch_size: int = 8,
        validation_split: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> dict:
        """
        Public method to create PyTorch DataLoaders directly from raw data.

        Args:
            data_matrix (pd.DataFrame): Raw data matrix with features as rows, samples as columns.
            design_matrix (pd.DataFrame): Contains group and sample identifiers.
            feature_column (str): Column name for feature identifiers in 'data_matrix'.
            group_column (str): Column indicating class/group labels.
            sample_column (str): Column indicating sample identifiers.
            batch_size (int): Batch size for DataLoader.
            validation_split (float): Fraction of data to use for validation. Set to 0 for no validation split.
            shuffle (bool): Whether to shuffle the data in DataLoader.
            random_state (int): RNG seed for reproducibility.

        Returns:
            dict: Contains 'train' DataLoader and optionally 'val' DataLoader.
        """
        # Align the data matrix to the network
        aligned_data = self._align_to_network(data_matrix, feature_column)

        # Prepare the data splits (train/val)
        data_splits = self._prepare_training_data(
            aligned_data,
            design_matrix,
            group_column,
            sample_column,
            validation_split,
            random_state,
        )

        # Create DataLoaders
        dataloaders = {
            "train": self._create_dataloader(*data_splits["train"], batch_size, shuffle)
        }
        if "val" in data_splits:
            dataloaders["val"] = self._create_dataloader(
                *data_splits["val"], batch_size, shuffle=False
            )
        return dataloaders

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """
        Internal method to create a PyTorch DataLoader from (X, y).

        Args:
            X (np.ndarray): Feature matrix (num_samples x num_features).
            y (np.ndarray): Target labels (num_samples,).
            batch_size (int): Batch size for DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: A PyTorch DataLoader.
        """
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)