from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

""" Class to store loggers """


class SuperLogger:
    def __init__(
        self,
        save_dir="",
        tensorboard=True,
        csv=True,
    ):
        self.tensorboard = tensorboard
        self.csv = csv
        self.logger_dict = {
            "csv": CSVLogger(save_dir),
            "tb": TensorBoardLogger(save_dir),
        }
        self.version = self.logger_dict["csv"].version

    def get_logger_list(self):
        l = []
        if self.tensorboard:
            l.append(self.logger_dict["tb"])
        if self.csv:
            l.append(self.logger_dict["csv"])
        return l
