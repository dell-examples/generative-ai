import yaml
import os
import logging
import sys
import torch

def run_command(command):
    """
    Execute a command in the system shell.

    Args:
        command (str): The command to be executed.

    Returns:
        None
    """
    os.system(command)

def validate_streams_yaml(yaml_file, logger):
    """
    Validates the format of a YAML configuration file.

    Parameters:
    yaml_file (str): File path of the YAML configuration.

    Returns:
    bool: True if the YAML format is valid, otherwise False.
    """
    if not os.path.isfile(yaml_file):
        logger.error(f"The file {yaml_file} does not exist.")
        sys.exit(1)

    try:
        with open(yaml_file, "r") as stream:
            data = yaml.safe_load(stream)

        if not isinstance(data, dict) or "rtsp_streams" not in data:
            return False

        gpus = data["gpu_config"]
        if not isinstance(gpus, dict):
            return False
        
        llm_gpus = gpus["llm_gpus"][0]
        if not isinstance(llm_gpus, int):
            return False

        return True

    except yaml.YAMLError:
        return False

if __name__ == '__main__':
    config_data = {}
    config_path = "/src/simulator_config.yaml"

    # Initialize logger
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # check if yaml file exists and is in correct format
    if validate_streams_yaml(config_path, logger):
        logger.info(f"The format of {config_path} is valid.")
    else:
        logger.error(f"The format of {config_path} is not valid.")
        sys.exit(1)

    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)

    gpu = config_data["gpu_config"]["llm_gpus"][0]

    avail_gpus = [i for i in range(torch.cuda.device_count())]

    if gpu not in avail_gpus:
        logger.error("LLM target GPU configuration is wrong! Correct it in simulator_config.yaml")
        sys.exit(1)

    command = f"CUDA_VISIBLE_DEVICES={gpu} uvicorn server:app --host 0.0.0.0 --port 8000"
    logger.info(command)

    run_command(command)
