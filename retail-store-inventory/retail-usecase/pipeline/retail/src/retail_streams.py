# Created by Scalers AI for Dell Inc.

import logging
import os
import sys
from multiprocessing import Pool

import fire
import yaml
from stream_processor import StreamProcessor


def start_inference(rtsp_url, key, broker, items_config, stream_name, logger):
    """
    Starts the inference process for a specific RTSP stream.

    Parameters:
    rtsp_url (str): The RTSP URL for video streaming.
    key (str): Key for the Zenoh publisher.
    broker (str): Zenoh broker address.
    items_config (str): File containing JSON data for zones and items.
    stream_name (str): Name of the video stream.
    """
    processor = StreamProcessor(
        rtsp_url, key, broker, items_config, stream_name, logger
    )
    processor.process_rtsp_video_stream()


def process_rtsp_streams(rtsp_streams, logger):
    """
    Processes multiple RTSP streams simultaneously.

    Parameters:
    rtsp_streams (dict): Dictionary containing configurations for RTSP streams.
    """
    pool_size = len(rtsp_streams)
    with Pool(processes=pool_size) as pool:
        args_list = []

        zenoh_port = 7447
        for stream_config in rtsp_streams:
            rtsp_url = rtsp_streams[stream_config]["url"]
            key = f"zenoh-pub-stream"
            broker = f"tcp/{rtsp_streams[stream_config]['broker']}:{str(zenoh_port)}"
            items_config = rtsp_streams[stream_config]["items_config"]

            args_list.append((rtsp_url, key, broker, items_config, stream_config, logger))
            zenoh_port += 1

        pool.starmap(start_inference, args_list)
        pool.close()
        pool.join()


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

        rtsp_streams = data["rtsp_streams"]
        if not isinstance(rtsp_streams, dict):
            return False

        for stream_name, stream_data in rtsp_streams.items():
            if (
                not isinstance(stream_data, dict)
                or "url" not in stream_data
                or "broker" not in stream_data
                or "items_config" not in stream_data
            ):
                return False

        return True

    except yaml.YAMLError:
        return False


def main(
    config: str,
):
    # Initialize logger
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # check if yaml file exists and is in correct format
    if validate_streams_yaml(config, logger):
        logger.info(f"The format of {config} is valid.")
    else:
        logger.error(f"The format of {config} is not valid.")
        sys.exit(1)

    with open(config, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        rtsp_streams = config_data.get("rtsp_streams", [])

    process_rtsp_streams(rtsp_streams, logger)


if __name__ == "__main__":
    fire.Fire(main)
