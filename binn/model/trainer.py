import torch
import torch.nn.functional as F


class BINNTrainer:
    """
    Handles training BINN models using a raw PyTorch training loop.
    """

    def __init__(self, binn_model, save_dir: str = ""):
        """
        Args:
            binn_model: The BINN model instance to train.
            save_dir (str): Directory to save logs or checkpoints.
        """
        self.save_dir = save_dir
        self.network = binn_model
        self.logger = BINNLogger(save_dir=save_dir)
        

    def fit(
        self,
        dataloaders: dict,
        num_epochs: int = 30,
        learning_rate: float = 1e-4,
        checkpoint_path: str = None,
    ):
        """
        Train the BINN model using a standard PyTorch training loop.

        Args:
            dataloaders (dict): Dictionary containing:
                - "train": DataLoader for training data.
                - "val" (optional): DataLoader for validation data.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            checkpoint_path (str): Path to save model checkpoints (optional).
        """

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):

            self.network.train()
            train_loss, train_accuracy = 0.0, 0.0

            for inputs, targets in dataloaders["train"]:
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)

                optimizer.zero_grad()
                outputs = self.network(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_accuracy += (torch.argmax(outputs, dim=1) == targets).float().mean().item()

            avg_train_loss = train_loss / len(dataloaders["train"])
            avg_train_accuracy = train_accuracy / len(dataloaders["train"])

            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}"
            )

            if "val" in dataloaders:
                self.network.eval()
                val_loss, val_accuracy = 0.0, 0.0
                with torch.no_grad():
                    for inputs, targets in dataloaders["val"]:
                        inputs = inputs.to(self.network.device)
                        targets = targets.to(self.network.device)

                        outputs = self.network(inputs)
                        loss = F.cross_entropy(outputs, targets)

                        val_loss += loss.item()
                        val_accuracy += (torch.argmax(outputs, dim=1) == targets).float().mean().item()

                avg_val_loss = val_loss / len(dataloaders["val"])
                avg_val_accuracy = val_accuracy / len(dataloaders["val"])

                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}"
                )


            if checkpoint_path:
                torch.save(self.network.state_dict(), f"{checkpoint_path}_epoch{epoch+1}.pt")

    def evaluate(self, dataloader):
        """
        Evaluate the BINN model on a dataset.

        Args:
            dataloader (DataLoader): DataLoader for the evaluation dataset.

        Returns:
            dict: A dictionary with 'loss' and 'accuracy' on the evaluation set.
        """
        self.network.eval()
        total_loss, total_accuracy = 0.0, 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.network.device)
                targets = targets.to(self.network.device)

                outputs = self.network(inputs)
                loss = F.cross_entropy(outputs, targets)

                total_loss += loss.item()
                total_accuracy += (torch.argmax(outputs, dim=1) == targets).float().mean().item()

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)

        return {"loss": avg_loss, "accuracy": avg_accuracy}
    
    def update_model(self, new_binn_model):
        self.binn_model = new_binn_model
        self.logger = BINNLogger(save_dir=self.save_dir)



class BINNLogger:
    """
    A minimal logger for BINN.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.logs = {"train": [], "val": []}

    def log(self, phase, metrics):
        """
        Log metrics for a specific phase (train/val).

        Args:
            phase (str): Phase name ('train' or 'val').
            metrics (dict): Dictionary containing metric names and values.
        """
        self.logs[phase].append(metrics)

    def save_logs(self):
        """
        Save logs to disk as a CSV file.
        """
        import pandas as pd
        for phase, log_data in self.logs.items():
            if log_data:
                df = pd.DataFrame(log_data)
                df.to_csv(f"{self.save_dir}/{phase}_logs.csv", index=False)
