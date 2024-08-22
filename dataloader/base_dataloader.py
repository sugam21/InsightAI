from abc import abstractmethod
import pandas as pd
from typing import Tuple, Dict
import os


class BaseDataLoader:
    def __init__(self, data_path: Dict[str, any]):
        self._train_dataset = None
        self._validation_dataset = None
        self._test_dataset = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None
        self.data_path: Dict[str, any] = data_path
        self._does_dir_exists()
        self._does_data_exists()

    def _does_dir_exists(self):
        """Checks if the data directory exists or not."""
        assert os.path.isdir(self.data_path['image_path']), f"{self.data_path['image_path']} directory does not exists."

    def _does_data_exists(self):
        """Checks if the data directory contains any data or not."""
        assert len(os.listdir(self.data_path['image_path'])) != 0, f"{self.data_path['image_path']} dir is empty."

    @abstractmethod
    def _get_splits(self, image_label_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Takes the pandas dataframe and splits into train test and validation dataframe
        Args:
            image_label_df (pd.DataFrame): The master dataframe containing images with its label.

        Returns:
            train, validation, test (Tuple): A tuple containing train, validation, and test dataframe.
        """
        ...

    @abstractmethod
    def get_train_dataloader(self, batch_size: int):
        """Returns the train dataloader with given batch size"""
        pass

    @abstractmethod
    def get_validataion_dataloader(self, batch_size: int):
        """Returns the validation dataloader with given batch size"""
        pass

    @abstractmethod
    def get_test_dataloader(self, batch_size: int):
        """Returns the test dataloader with given batch size"""
        pass
