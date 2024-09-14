import torch
from tqdm import tqdm

from src.base import BaseTrainer
from logger import get_logger

LOG = get_logger("training")


class Trainer(BaseTrainer):

    def __init__(
        self,
        model: any,
        config,
        device,
        train_dataloader,
        valid_dataloader=None,
        lr_scheduler=None,
    ) -> None:
        super().__init__(model, config)
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(train_dataloader.batch_size
                           )  # either do this or use batch size from config.

    def _train_epoch(self, epoch: int):
        """Train logic for 1 epoch.
        Args:
            epoch (int): Current epoch.
        Returns:
            result (dict[str, any]): A log containing average loss and metric of epoch.
        """
        self.model.train()
        total_loss: int = 0
        total_accuracy: int = 0
        num_batch: int = len(self.train_dataloader)
        result: dict[str, any] = {}
        for batch_idx, (data, target) in tqdm(enumerate(self.train_dataloader),
                                              unit="batch",
                                              total=num_batch):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(
                data)  # this is softmax prob. in size [batch, 17]
            loss = self.criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            accuracy: float = self.metric(output, target)
            total_accuracy += accuracy
            if batch_idx % self.log_step == 0:
                LOG.debug(
                    "Epoch: {} Batch: {} Loss: {:.4f} Accuracy: {:.4f}".format(
                        epoch, batch_idx + 1, loss.item(), accuracy))
        result["train_loss"] = total_loss / num_batch
        result["train_accuracy"] = total_accuracy / num_batch

        if self.valid_dataloader is not None:
            val_log = self._valid_epoch(epoch)
            result.update(val_log)

        return result

    def _valid_epoch(self, epoch: int):
        """Train logic for validation data for 1 epoch."""
        self.model.eval()
        total_loss: int = 0
        total_accuracy: int = 0
        val_log: dict[str, any] = {}
        num_batch: int = len(self.valid_dataloader)
        with torch.no_grad():
            for batch_idx, (data,
                            target) in tqdm(enumerate(self.valid_dataloader),
                                            unit="batch",
                                            total=num_batch):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(
                    data)  # this is softmax prob. in size [batch, 17]
                loss = self.criterion(output, target)
                accuracy: float = self.metric(output, target)
                total_loss += loss.item()
                total_accuracy += accuracy
        val_log["val_loss"] = total_loss / num_batch
        val_log["val_accuracy"] = total_accuracy / num_batch
        return val_log
