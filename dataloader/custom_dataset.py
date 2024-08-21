from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import Tensor
import pandas as pd
import os


class DataTransform:
    def __init__(self, input_size: int, channel_mean, channel_std):
        self.data_transform = {
            "base": A.Compose(
                [
                    A.RandomCrop(width=input_size, height=input_size),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=channel_mean, std=channel_std, max_pixel_value=255),
                    ToTensorV2(),
                ]
            ),
            "train": A.Compose(
                [
                    A.RandomCrop(width=input_size, height=input_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=1.0),
                    A.ShiftScaleRotate(p=0.2),
                    A.Normalize(mean=channel_mean, std=channel_std, max_pixel_value=255),
                    ToTensorV2(),
                ]
            ),
            "valid": A.Compose(
                [
                    A.Normalize(mean=channel_mean, std=channel_std, max_pixel_value=255),
                    ToTensorV2(),
                ]
            ),
        }

    def __call__(self, image: Tensor, is_train: bool = False, is_augment: bool = False):
        transform_type: str
        if is_train:
            transform_type = "train" if is_augment else "base"
        else:
            transform_type = "base"
        transformed = self.data_transform[transform_type](image=image)
        return transformed['image']


class CustomDataset(Dataset):
    def __init__(self, image_data_path: str, image_label_df: pd.DataFrame, is_train: bool = False,
                 transform: any = None):
        self.image_data_path: str = image_data_path
        self.is_train: bool = is_train
        self.image_label_df: pd.DataFrame = pd.DataFrame()
        if self.is_train:
            temp_image_label_df: pd.DataFrame = image_label_df
            temp_image_label_df['alteration'] = False  # added a new column
            copy_image_label_df: pd.DataFrame = temp_image_label_df.copy()
            copy_image_label_df['alteration'] = True  # these are the images to augment
            self.image_label_df = pd.concat([temp_image_label_df, copy_image_label_df],
                                            axis=0).sample(frac=1)
        else:
            self.image_label_df = image_label_df
            self.image_label_df['alteration'] = False
        self.transform: any = transform

    def len(self) -> int:
        return len(self.image_label_df)

    def __getitem__(self, idx: int):
        image_file_name: str = self.image_label_df.iloc[idx, 0]
        image_file_label_str: str = self.image_label_df.iloc[idx, 1]
        image_file_label_int: int = int(image_file_label_str[1:])
        is_augment: bool = self.image_label_df.iloc[idx, 2]
        full_image_file_path: str = os.path.join(self.image_data_path, image_file_label_str, image_file_name)
        image_tensor: np.array = np.array(Image.open(full_image_file_path).convert("RGB"))
        if self.transform:
            image_tensor = self.transform(image=image_tensor, is_train=self.is_train, is_augment=is_augment)
        return image_tensor, image_file_label_int
