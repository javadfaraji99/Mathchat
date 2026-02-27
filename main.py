import os
from pesudo_main import pseudo_main
from flaml import oai
import json


def load_config(config_file="OAI_CONFIG_LIST.json"):
    """Load config_list from a JSON file."""
    with open(config_file, "r") as f:
        config_list = json.load(f)
    return config_list


def main():
    pseudo_main(config_list)


if __name__ == "__main__":
    # Load OpenAI config from JSON
    config_list = load_config("OAI_CONFIG_LIST.json")
    
    os.environ["WOLFRAM_ALPHA_APPID"] = open("wolfram.txt").read().strip()
    oai.retry_timeout = 3600
    main()