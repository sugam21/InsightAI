from .custom_dataset import CustomDataset, DataTransform
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class CustomDataLoader:
    def __init__(self, data_path: Dict[str, any]):
        self.test_dataloader = None
        self.test_dataset = None
        self.validation_dataloader = None
        self.validation_dataset = None
        self.train_dataloader = None
        self.train_dataset = None
        self.data_path: Dict[str, any] = data_path

        self.image_label_df: pd.DataFrame = pd.read_csv(data_path['image_labels_path']).drop(columns=['Unnamed: 0'],
                                                                                             axis=1)
        self.train, self.validation, self.test = self._get_splits(self.image_label_df)
        self.transformation_params: Dict[str, any] = {"input_size": self.data_path['image_size'],
                                                      "channel_mean": self.data_path['channel_mean'],
                                                      "channel_std": self.data_path['channel_std']}

    def _get_splits(self, image_label_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Takes the image-label dataframe and splits into training, validation and testing set
        with ratio of 6:2:2"""

        train, temp_test = train_test_split(image_label_df, train_size=0.6,
                                            random_state=self.data_path['random_state'],
                                            stratify=image_label_df['class_label'])
        validation, test = train_test_split(temp_test, train_size=0.5,
                                            random_state=self.data_path['random_state'], )
        return train, validation, test

    def get_train_dataloader(self, batch_size: int = 32):
        self.train_dataset = CustomDataset(image_data_path=self.data_path['image_path'],
                                           image_label_df=self.train,
                                           is_train=True,
                                           transform=DataTransform(**self.transformation_params)
                                           )

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        return self.train_dataloader

    def get_validataion_dataloader(self, batch_size: int = 32):
        self.validation_dataset = CustomDataset(image_data_path=self.data_path['image_path'],
                                                image_label_df=self.validation,
                                                is_train=False,
                                                # transform=DataTransform(input_size=self.data_path['image_size'],
                                                #                         channel_mean=self.data_path['channel_mean'],
                                                #                         channel_std=self.data_path['channel_std'])
                                                transform=DataTransform(**self.transformation_params))

        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size)
        return self.validation_dataloader

    def get_test_dataloader(self, batch_size: int = 32):
        self.test_dataset = CustomDataset(image_data_path=self.data_path['image_path'],
                                          image_label_df=self.test,
                                          is_train=False,
                                          transform=DataTransform(**self.transformation_params),)

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size)
        return self.test_dataloader
