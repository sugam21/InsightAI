import argparse

import torch

from src.dataloader import CustomDataLoader
from src.model import MobileNet
from src.trainer import Trainer
from src.utils import Config, seed_everything
from src.utils import get_logger

CONFIG_PATH: str = r"../../config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG = get_logger(__name__)


def main() -> None:
    config: Config = Config.from_json(CONFIG_PATH)
    LOG.debug("Configuration Loaded successfully. ✔")

    seed_everything(config.train["seed"])

    dataloader = CustomDataLoader(config.data)
    train_dataloader = dataloader.get_train_dataloader(
        batch_size=config.train["batch_size"])
    validation_dataloader = dataloader.get_validation_dataloader(
        batch_size=config.train["batch_size"])

    # visualize_image(validation_dataloader, num_batch_to_show=1)

    model = MobileNet(config.model["output"])

    trainer = Trainer(
        model=model,
        config=config.train,
        train_dataloader=train_dataloader,
        valid_dataloader=validation_dataloader,
        device=DEVICE,
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-s",
        "--save",
        default=False,
        type=str,
        help="whether to save the model or not.",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint. default(None)",
    )

    main()
