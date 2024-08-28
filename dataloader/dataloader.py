from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from base.base_dataloader import BaseDataLoader
from logger import get_logger

from .custom_dataset import CustomDataset
from .data_transform import DataTransform

logger: any = get_logger("dataloader")


class CustomDataLoader(BaseDataLoader):
    def __init__(self, data_path: Dict[str, any]):
        super().__init__(data_path)

        logger.debug(f"Loading the data from {self.data_path['image_path']}")
        logger.info(f"Loading the data from {self.data_path['image_path']}")

        self.image_label_df: pd.DataFrame = pd.read_csv(
            self.data_path["image_labels_path"]
        ).drop(columns=["Unnamed: 0"], axis=1)
        self.train, self.validation, self.test = self._get_splits(self.image_label_df)

    def _get_splits(
        self, image_label_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_size: float = 0.8
        test_size: float = (1 - train_size) / 2
        validation_size: float = (1 - train_size) / 2
        logger.warning(
            f"Splitting the data into training: {train_size}, validation: {validation_size} and testing: {test_size}"
        )
        train, temp_test = train_test_split(
            image_label_df,
            train_size=train_size,
            random_state=self.data_path["random_state"],
            stratify=image_label_df["class_label"],
        )
        validation, test = train_test_split(
            temp_test,
            train_size=0.5,
            random_state=self.data_path["random_state"],
        )
        return train, validation, test

    def get_train_dataloader(self, batch_size: int = 32):
        self._train_dataset = CustomDataset(
            image_data_path=self.data_path["image_path"],
            image_label_df=self.train,
            is_train=True,
            transform=DataTransform(input_size=self.data_path["image_size"]),
        )

        self.train_dataloader = DataLoader(
            self._train_dataset, batch_size=batch_size, shuffle=True
        )
        return self.train_dataloader

    def get_validation_dataloader(self, batch_size: int = 32):
        self._validation_dataset = CustomDataset(
            image_data_path=self.data_path["image_path"],
            image_label_df=self.validation,
            is_train=False,
            transform=DataTransform(input_size=self.data_path["image_size"]),
        )

        self.validation_dataloader = DataLoader(
            self._validation_dataset, batch_size=batch_size
        )
        return self.validation_dataloader

    def get_test_dataloader(self, batch_size: int = 32):
        self._test_dataset = CustomDataset(
            image_data_path=self.data_path["image_path"],
            image_label_df=self.test,
            is_train=False,
            transform=DataTransform(input_size=self.data_path["image_size"]),
        )

        self.test_dataloader = DataLoader(self._test_dataset, batch_size=batch_size)
        return self.test_dataloader
