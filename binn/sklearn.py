from typing import Union

import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from sklearn.base import BaseEstimator, ClassifierMixin

from binn import BINN, Network, SuperLogger


class BINNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pathways: Network = None,
        activation: str = "tanh",
        weight: torch.Tensor = torch.Tensor([1, 1]),
        learning_rate: float = 1e-4,
        n_layers: int = 4,
        scheduler="plateau",
        optimizer="adam",
        validate: bool = True,
        n_outputs: int = 2,
        dropout: float = 0,
        residual: bool = False,
        threads: int = 1,
        epochs: int = 100,
        logger: Union[SuperLogger, None] = None,
        log_steps: int = 50
    ):
        self.clf = BINN(
            pathways=pathways,
            activation=activation,
            weight=weight,
            learning_rate=learning_rate,
            n_layers=n_layers,
            scheduler=scheduler,
            optimizer=optimizer,
            validate=validate,
            n_outputs=n_outputs,
            dropout=dropout,
            residual=residual,
        )

        self.threads = threads
        self.epochs = epochs
        self.logger = logger
        self.log_steps = log_steps

    def fit(self, X, y):
        dataloader = DataLoader(
            dataset=TensorDataset(torch.Tensor(X), torch.LongTensor(y)),
            batch_size=8,
            num_workers=self.threads,
            shuffle=True,
        )

        trainer = Trainer(
            callbacks=[], logger=self.logger.get_logger_list(), max_epochs=self.epochs,
            log_every_n_steps=self.log_steps
        )

        trainer.fit(self.clf, dataloader)

    def predict(self, X):

        return NotImplemented

    @property
    def feature_importances_(self):

        return NotImplemented
