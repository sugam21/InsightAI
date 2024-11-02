import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        img_data_path: str,
        img_label_df: pd.DataFrame,
        is_train: bool = False,
        transform=None,
    ):
        self.img_data_path: str = img_data_path
        self.is_train: bool = is_train
        self.img_label_df = img_label_df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_label_df)

    def __getitem__(self, idx: int):
        indexed_row = self.img_label_df.iloc[idx]
        img_file_name = indexed_row["img"]
        img_file_label_str = indexed_row["class"]
        img_file_label_int = int(img_file_label_str[1:])

        img_file_path = os.path.join(
            self.img_data_path, img_file_label_str, img_file_name
        )
        img_array = np.array(Image.open(img_file_path).convert("RGB"))
        if self.transform:
            img_tensor = self.transform(img_array)

        return img_tensor, img_file_label_int
