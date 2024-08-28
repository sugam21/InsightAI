import json
import logging.config
from typing import Dict
import os


def setup_logging(save_dir: str, log_config_file_name: str = r"logger/logging_configs.json", default_level=logging.INFO):
    """Setup logging configuration
    Args:
        save_dir (str): directory to save logs. This is the path(log_save_dir) present in your main config.json.
        log_config_file_name (str): path to config file.
        default_level (str): default level to log
    Returns:
        None
    """
    if os.path.isfile(log_config_file_name):
        with open(log_config_file_name, mode='r') as f:
            log_config: Dict[str, any] = json.load(f)

        for _, handler in log_config["handlers"].items():
            # Updates the filename in logging_config with the full path.
            if "filename" in handler:
                handler['filename'] = os.path.join(save_dir, handler['filename'])
        logging.config.dictConfig(log_config)
    else:
        # print(f"Warning: ")
        logging.warning(f"logging configuration file is missing from {log_config_file_name}.")
        logging.basicConfig(level=default_level)


def get_logger(name: str):
    """Get logger with provided name.
    Args:
        name (str): name of the logger.
    Returns:
        custom_logger
    """
    logger: any = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
