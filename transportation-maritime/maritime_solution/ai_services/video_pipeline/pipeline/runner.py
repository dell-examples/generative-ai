# Created by Scalers AI for Dell Inc.

import yaml
import os
import concurrent.futures
import logging
import sys
import torch

def divide_dict(dictionary, n):
    """
    Divide a dictionary into approximately equal-sized divisions.

    Args:
        dictionary (dict): The dictionary to be divided.
        n (int): The number of divisions to create.

    Returns:
        tuple: A tuple containing:
            - divisions (list): A list of dictionaries representing the divisions.
            - divisions_sizes (list): A list containing the number of items in each division.
    """
    items = list(dictionary.items())
    total_items = len(items)
    
    # Calculate the number of items in each division
    items_per_division = total_items // n
    remainder = total_items % n
    
    divisions = []
    divisons_sizes = []
    start_index = 0
    
    # Create divisions
    for i in range(n):
        division_size = items_per_division + (1 if i < remainder else 0)
        divisons_sizes.append(division_size)
        divisions.append(dict(items[start_index:start_index + division_size]))
        start_index += division_size
    
    return divisions, divisons_sizes

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
        
        video_gpus = gpus["video_pipeline_gpus"]
        if not isinstance(video_gpus, list):
            return False

        rtsp_streams = data["rtsp_streams"]
        if not isinstance(rtsp_streams, dict):
            return False

        for stream_name, stream_data in rtsp_streams.items():
            if (
                not isinstance(stream_data, dict)
                or "url" not in stream_data
                or "broker" not in stream_data
                or "zone" not in stream_data
                or "visualize" not in stream_data
            ):
                return False

        return True

    except yaml.YAMLError:
        return False

if __name__ == '__main__':
    config_data = {}
    config_path = "/pipeline/config.yml"

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

    gpus = config_data["gpu_config"]["video_pipeline_gpus"]
    streams = config_data["rtsp_streams"]

    avail_gpus = [i for i in range(torch.cuda.device_count())]

    if all(item in avail_gpus for item in gpus) == False:
        logger.error("Video Pipeline GPU configured is wrong! Please correct it in simulator_config.yaml")
        sys.exit(1)

    divisions, sizes = divide_dict(streams, len(gpus))

    for stream, gpu_id in zip(divisions, gpus):
        final_dict = {'rtsp_streams': {}}

        for key, value in stream.items():
            final_dict['rtsp_streams'][key] = value
            
        with open(f"gpu{gpu_id}.yaml", 'w') as yaml_file:
            yaml.dump(final_dict, yaml_file, default_flow_style=False)

    commands = []
    udp_port = 1234
    zenoh_port = 7447

    for gpu_id, size in zip(gpus, sizes):
        commands.append(f"CUDA_VISIBLE_DEVICES={gpu_id} python3 yolo_pipeline.py -i gpu{gpu_id}.yaml -z {zenoh_port} -u {udp_port}")
        zenoh_port += size
        udp_port += size

    # Using ThreadPoolExecutor to run commands concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each command to the executor
        futures = [executor.submit(run_command, cmd) for cmd in commands]

        # Wait for all commands to complete
        concurrent.futures.wait(futures)


    





