from utils.utils import Config, visualize_image, seed_everything
from dataloader import CustomDataLoader

CONFIG_PATH: str = r"config.json"


def main() -> None:
    config: Config = Config.from_json(CONFIG_PATH)

    seed_everything(config.train['seed'])

    dataloader = CustomDataLoader(config.data)
    train_dataloader = dataloader.get_train_dataloader(batch_size=config.train['batch_size'])
    validation_dataloader = dataloader.get_validation_dataloader(batch_size=config.train['batch_size'])
    test_dataloader = dataloader.get_test_dataloader(batch_size=config.train['batch_size'])

    visualize_image(validation_dataloader, num_batch_to_show=1)


if __name__ == "__main__":
    main()
