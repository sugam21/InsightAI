from abc import abstractmethod
# from typing import Dict
import os
from src.utils import check_dir_if_exists
from logger import get_logger
import torch
import logging
import torch.optim
from src import model as model_loss
import src.model.metric as model_metric

LOG = get_logger("trainer")


class BaseTrainer:

    def __init__(self, model, config) -> None:
        self.model: any = model
        self.config_train: dict[str, any] = config

        # self.optimizer = optimizer
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        self.metric: any = self._get_metric()

        self.epochs: int = self.config_train["epochs"]
        self.save_period: int = self.config_train["save_period"]
        self.start_epoch: int = 1
        self.checkpoint_dir = self.config_train["checkpoint_save_dir"]

        check_dir_if_exists(self.checkpoint_dir)

        if self.config_train["resume"] != "":
            self._resume_checkpoint(self.config_train.get("resume"))

    @abstractmethod
    def _train_epoch(self, epoch: int):
        """Training logic for an epoch
        Args:
            epoch (int): number of epoch
        """
        raise NotImplementedError

    def _get_criterion(self):
        return getattr(model_loss, self.config_train["loss"])

    def _get_metric(self):
        return getattr(model_metric, self.config_train["metric"])

    def _get_optimizer(self):
        module_name: str = self.config_train["optimizer"]["type"]
        module_params: dict[str, any] = dict(self.config_train["optimizer"]["args"])
        return getattr(torch.optim, module_name)(self.model.parameters(),
                                                 **module_params)

    def train(self):
        """Complete training epoch"""
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            LOG.info("Epoch: {}".format(epoch))

            log: dict[str, float] = {}
            log.update(result)
            # add logging here
            for key, value in log.items():
                LOG.info("{}:{:.3f}".format(str(key), value))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int) -> None:
        """Saves the checkpoint in the specified dir."""
        LOG.info("----Saving Checkpoint----")
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config_train,
        }

        filename: str = os.path.join(self.checkpoint_dir,
                                     f"checkpoint-epoch{epoch}.pth")
        torch.save(state, filename)
        logging.info(f"Saving checkpoint: {filename}.......")

    def _resume_checkpoint(self, resume_path: str) -> None:
        """Resume Checkpoint using the resume_path file."""
        resume_path: str = str(resume_path)
        LOG.info(f"Loading checkpoint: {resume_path}.....")
        checkpoint = torch.load(resume_path, weights_only=False)
        self.start_epoch: int = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        LOG.info(
            f"Checkpoints loaded. Resume training from epoch {self.start_epoch}"
        )
