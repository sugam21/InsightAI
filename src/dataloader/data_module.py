import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from .custom_dataset import CustomDataset


class LaptopDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, img_df, config):
        super().__init__()
        self.batch_size = batch_size
        self.img_df = img_df
        self.config = config

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train, valid = train_test_split(
                self.img_df,
                train_size=self.config["train_split"],
                random_state=self.config["seed"],
                stratify=self.img_df["class"],
            )
            self.train_dataset = CustomDataset(
                img_data_path=self.config["image_path"],
                img_label_df=train,
                is_train=True,
                transform=self.transform,
            )

            self.valid_dataset = CustomDataset(
                img_data_path=self.config["image_path"],
                img_label_df=valid,
                is_train=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)
