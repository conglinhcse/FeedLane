import json


def parse_configs(config_file):
    """Configuration parsing

    Args:
        config_file (str): configuration file path

    Returns:
        dict: dict of configurations
    """
    with open(config_file, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    print(parse_configs("config/nn.json"))