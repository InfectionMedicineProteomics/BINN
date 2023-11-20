from lightning.pytorch.loggers import  CSVLogger


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
        }
        self.version = self.logger_dict["csv"].version
        self.save_dir = save_dir

    def get_logger_list(self):
        loggers = []
        if self.csv:
            loggers.append(self.logger_dict["csv"])
        return loggers
