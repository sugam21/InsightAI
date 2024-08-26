from abc import abstractmethod
from typing import Dict
import os


class BaseTrainer:
    def __init__(self, model, config) -> None:
        self.model: any = model
        self.config_train: Dict[str, any] = config
        self.optimizer: any = self.config_train['optimizer']
        self.loss: any = self.config_train['loss']
        # self.criterion: any = criterion
        self.epochs: int = self.config_train['epochs']
        self.save_period: int = self.config_train['save_period']
        self.start_epoch: int = 1
        self.checkpoint_dir = self.config_train['checkpoint_save_dir']
        self._check_if_exists()
        if self.config_train.get("resume") is not None:
            self._resume_checkpoint(self.config_train.get("resume"))

    def _check_if_exists(self):
        """Check if checkpoint saving directory exists or not."""
        assert os.path.isdir(self.checkpoint_dir), f"{self.checkpoint_dir} does not exists."

    @abstractmethod
    def _train_epoch(self, epoch: int):
        """Training logic for an epoch
        Args:
            epoch (int): number of epoch
        """
        raise NotImplementedError

    def train(self):
        """Complete training epoch"""
        not_improved_count: int = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {"epoch": epoch}
            log.update(result)
            # add logging here

    def _save_checkpoint(self):
        ...

    def _resume_checkpoint(self, resume_path: str):
        ...
