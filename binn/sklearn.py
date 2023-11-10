from typing import Union
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from sklearn.base import BaseEstimator, ClassifierMixin

from binn import BINN, Network, SuperLogger


class BINNClassifier(BaseEstimator, ClassifierMixin):
    """
    A sci-kit learn wrapper for the BINN.

    Args:

        pathways : Network, optional
            The network architecture to use for the classifier. If None, a default
            architecture will be used. Default is None.
        activation : str, optional
            The activation function to use for the classifier. Default is 'tanh'.
        weight : torch.Tensor, optional
            The weight to assign to each class. Default is torch.Tensor([1, 1]).
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 1e-4.
        n_layers : int, optional
            The number of layers in the network architecture. Default is 4.
        scheduler : str, optional
            The scheduler to use for the optimizer. Default is 'plateau'.
        optimizer : str, optional
            The optimizer to use for training. Default is 'adam'.
        n_outputs : int, optional
            The number of outputs of the network architecture. Default is 2.
        dropout : float, optional
            The dropout rate to use for the classifier. Default is 0.
        residual : bool, optional
            Whether to use residual connections in the network architecture.
            Default is False.
        threads : int, optional
            The number of threads to use for data loading. Default is 1.
        epochs : int, optional
            The number of epochs to train the classifier for. Default is 100.
        logger : Union[SuperLogger, None], optional
            The logger to use for logging training information. Default is None.
        log_steps : int, optional
            The number of steps between each log message during training.
            Default is 50.

    Attributes:
        clf : BINN
            The BINN (Block Independent Neural Network) instance used for
            classification.
        threads : int
            The number of threads used for data loading.
        epochs : int
            The number of epochs to train the classifier for.
        logger : Union[SuperLogger, None]
            The logger used for logging training information.
        log_steps : int
            The number of steps between each log message during training.

    """

    def __init__(
        self,
        network: Network = None,
        activation: str = "tanh",
        weight: torch.Tensor = torch.Tensor([1, 1]),
        learning_rate: float = 1e-4,
        n_layers: int = 4,
        scheduler: str = "plateau",
        optimizer: str = "adam",
        n_outputs: int = 2,
        dropout: float = 0,
        residual: bool = False,
        threads: int = 1,
        epochs: int = 100,
        logger: Union[SuperLogger, None] = None,
        log_steps: int = 50,
    ):
        self.clf = BINN(
            network=network,
            activation=activation,
            weight=weight,
            learning_rate=learning_rate,
            n_layers=n_layers,
            scheduler=scheduler,
            optimizer=optimizer,
            validate=False,
            n_outputs=n_outputs,
            dropout=dropout,
            residual=residual,
        )

        self.threads = threads
        self.epochs = epochs
        self.logger = logger
        self.log_steps = log_steps

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int):
        """
        Trains the classifier using the provided input data and target labels.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data.
            y (array-like of shape (n_samples,)): The target labels.

        Returns:
            None
        """
        if epochs != None:
            self.epochs = epochs

        dataloader = DataLoader(
            dataset=TensorDataset(torch.Tensor(X), torch.LongTensor(y)),
            batch_size=8,
            num_workers=self.threads,
            shuffle=True,
        )

        trainer = pl.Trainer(
            callbacks=[],
            logger=self.logger.get_logger_list(),
            max_epochs=self.epochs,
            log_every_n_steps=self.log_steps,
        )

        trainer.fit(self.clf, dataloader)

    def predict(self, X: np.ndarray):
        """
        Predicts target labels for the provided input data using the trained classifier.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data.

        Returns:
            y_hat (torch.Tensor of shape (n_samples,)): The predicted target labels.
        """
        X = torch.Tensor(X)
        with torch.no_grad():
            y_hat = self.clf(X)
        return y_hat
