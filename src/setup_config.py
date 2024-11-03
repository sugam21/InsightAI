import json
import tomllib
from pathlib import Path


class Config:
    """Takes the config file path and returns a Config object while setting up logger."""

    def __init__(self, data: dict[str, any], train: dict[str, any]) -> None:
        self.data: dict[str, any] = data
        self.train: dict[str, any] = train

    @classmethod
    def from_json(cls, config_path: Path | str):
        with open(config_path, mode="r") as file:
            config: dict[str, any] = json.load(file)
        return cls(config["data"], config["train"])

    @classmethod
    def from_toml(cls, config_path: Path | str):
        with open(config_path, mode="rb") as file:
            config: dict[str, any] = tomllib.load(file)
        return cls(config["data"], config["train"])
