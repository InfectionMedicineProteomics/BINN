from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


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
        self.save_dir = save_dir

    def get_logger_list(self):
        loggers = []
        if self.tensorboard:
            loggers.append(self.logger_dict["tb"])
        if self.csv:
            loggers.append(self.logger_dict["csv"])
        return loggers
