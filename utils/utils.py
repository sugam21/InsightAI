import json
from typing import Dict


class Config:
    """Takes the config file path and returns a Config object."""
    def __init__(self, data: Dict[str, any], train: Dict[str, any], model: Dict[str, any]) -> None:
        self.data: Dict[str, any] = data
        self.train: Dict[str, any] = train
        self.model: Dict[str, any] = model

    @classmethod
    def from_json(cls, config_path):
        with open(config_path, mode='r') as file:
            config: Dict[str, any] = json.load(file)
        return cls(config['data'], config['train'], config['model'])
