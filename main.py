from utils.utils import Config
from dataloader import CustomDataLoader
import matplotlib.pyplot as plt


CONFIG_PATH: str = r"config.json"


def visualize_image(loader):
    for i in range(10):
        try:
            train_features, train_labels = next(iter(loader))
        except Exception as e:
            print('------------exception-----------')
        else:
            print(f"Feature batch shape: {train_features.size()}")
            print(f"Labels batch shape: {train_labels.size()}")
            img = train_features[31].permute(1, 2, 0)
            label = train_labels[0]
            plt.imshow(img)
            plt.show()
            print(f"Label: {label}")

def main() -> None:
    """TODO: valid and test dataloader printing same image again and again.
    make sure to uncomment valid normalization part in custom dataset transformation."""
    config: Config = Config.from_json(CONFIG_PATH)
    dataloader = CustomDataLoader(config.data)
    train_dataloader = dataloader.get_train_dataloader(batch_size=config.train['batch_size'])
    validation_dataloader = dataloader.get_validataion_dataloader(batch_size=config.train['batch_size'])
    test_dataloader = dataloader.get_test_dataloader(batch_size=config.train['batch_size'])

    visualize_image(train_dataloader)


if __name__ == "__main__":
    main()
