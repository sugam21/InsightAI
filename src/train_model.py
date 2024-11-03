from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import wandb
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

from src import Config
from src.dataloader import ImagePredictionLogger, LaptopDataModule
from src.model import LitModel

CONFIG_PATH = Path("config.json").resolve()


def fix_path(config):
    config.data["image_path"] = Path(config.data["image_path"]).resolve()
    config.data["image_labels_path"] = Path(config.data["image_labels_path"]).resolve()
    config.train["checkpoint_save_dir"] = Path(
        config.train["checkpoint_save_dir"]
    ).resolve()
    return config


def main() -> None:
    config: Config = Config.from_json(CONFIG_PATH)
    config = fix_path(config)
    logger.info("Configuration Loaded successfully.")

    wandb_logger = WandbLogger(
        project="insight_ai_custom_model", job_type="train", id="base_model"
    )

    pl.seed_everything(config.train["seed"])

    df = pd.read_csv(config.data["image_labels_path"]).drop(
        columns=["Unnamed: 0"], axis=1
    )
    dm = LaptopDataModule(
        batch_size=config.train["batch_size"], img_df=df, config=config.data
    )
    # To access the x_dataloader we need to call prepare_data and setup.
    dm.prepare_data()
    dm.setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    logger.info(f"Validation Image shape {val_imgs.shape}")
    logger.info(f"Validation Image Labels shape {val_labels.shape}")

    model = LitModel(input_shape=(3, 224, 224), num_classes=config.train["num_class"])

    # Initialize Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.train["checkpoint_save_dir"]
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=15,
        logger=wandb_logger,
        callbacks=[
            early_stop_callback,
            ImagePredictionLogger(val_samples),
            checkpoint_callback,
        ],
        accelerator="auto",
        devices=1,
        fast_dev_run=5,
    )

    # Train the model ⚡🚅⚡
    trainer.fit(model, dm)

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    # Initialize wandb logger
    main()
