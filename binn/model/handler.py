from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
import torch.nn.functional as F


class BINNHandler:
    def __init__(self, network, save_dir="", csv=True, tensorboard=False,):
        """
        Handles data alignment, preparation, and training for the BINN model, with integrated logging.

        Args:
            network: The BINN network instance.
            save_dir (str): Directory to save logs.
            tensorboard (bool): Whether to use TensorBoard logging.
            csv (bool): Whether to use CSV logging.
        """
        self.network = network
        self.super_logger = SuperLogger(save_dir=save_dir, tensorboard=tensorboard, csv=csv)

    def align_to_network(self, data_matrix, feature_column="Protein"):
        """
        Align the input data matrix to the features expected by the BINN network.

        Args:
            data_matrix (pd.DataFrame): Input data matrix.
            feature_column (str): The column containing feature identifiers.

        Returns:
            pd.DataFrame: Aligned data matrix.
        """
        features = self.network.inputs
        dm = data_matrix.copy()

        # Handle missing features
        if len(features) > len(dm.index):
            features_df = pd.DataFrame(features, columns=[feature_column])
            dm = dm.merge(features_df, how="right", on=feature_column)

        # Reindex to match the feature order
        dm.set_index(feature_column, inplace=True)
        dm = dm.loc[features].fillna(0)
        return dm

    def prepare_training_data(self, data_matrix, design_matrix, group_column="group", sample_column="sample"):
        """
        Prepare the training data (X, y) from the data matrix and design matrix.

        Args:
            data_matrix (pd.DataFrame): Aligned data matrix.
            design_matrix (pd.DataFrame): Design matrix with group and sample information.
            group_column (str): Column indicating the group (e.g., labels).
            sample_column (str): Column indicating the sample identifiers.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target labels.
        """
        # Identify groups
        groups = design_matrix[group_column].unique()
        group_samples = {
            group: design_matrix[design_matrix[group_column] == group][sample_column].values.tolist()
            for group in groups
        }

        # Create X and y
        X_list = [data_matrix[samples].T for samples in group_samples.values()]
        X = pd.concat(X_list).fillna(0).to_numpy()
        y = np.concatenate([[i] * len(samples) for i, samples in enumerate(group_samples.values())])

        # Standardize features
        X = preprocessing.StandardScaler().fit_transform(X)
        return X, y

    def create_dataloader(self, X, y, batch_size=8, shuffle=True):
        """
        Create a PyTorch DataLoader from the prepared training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: PyTorch DataLoader.
        """
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_binn(self, dataloader, num_epochs=30):
        """
        Train the BINN using a standard PyTorch training loop.

        Args:
            dataloader (DataLoader): DataLoader containing training data.
            num_epochs (int): Number of training epochs.
        """
        optimizer = self.network.configure_optimizers()[0][0]

        for epoch in range(num_epochs):
            self.network.train()
            total_loss = 0.0
            total_accuracy = 0

            for inputs, targets in dataloader:
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)

                optimizer.zero_grad()
                outputs = self.network(inputs).to(self.network.device)

                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_accuracy += torch.sum(torch.argmax(outputs, axis=1) == targets).item() / len(targets)

            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)
            print(f"Epoch {epoch}: Avg Accuracy: {avg_accuracy:.4f}, Avg Loss: {avg_loss:.4f}")

    def train_binn_with_lightning(self, dataloader, max_epochs=10):
        """
        Train the BINN using Lightning Trainer.

        Args:
            dataloader (DataLoader): DataLoader containing training data.
            max_epochs (int): Number of training epochs.
        """
        trainer = pl.Trainer(
            logger=self.super_logger.get_logger_list(),
            max_epochs=max_epochs,
            log_every_n_steps=10,
        )
        trainer.fit(self.network, dataloader)


class SuperLogger:
    """
    A unified logger for BINN, supporting CSV and TensorBoard logging.
    """

    def __init__(self, save_dir="", tensorboard=True, csv=True):
        self.tensorboard = tensorboard
        self.csv = csv
        self.logger_dict = {
            "csv": CSVLogger(save_dir),
        }
        self.version = self.logger_dict["csv"].version
        self.save_dir = save_dir

    def get_logger_list(self):
        """
        Get the list of active loggers.

        Returns:
            list: A list of active loggers.
        """
        loggers = []
        if self.csv:
            loggers.append(self.logger_dict["csv"])
        return loggers
