# Created by Scalers AI for Dell Inc.

import json
import logging
import os
import sys
import time
from multiprocessing import Pool

import ffmpeg
import fire
import numpy as np
import yaml
import zenoh
from zenoh import Reliability, Sample


class Reciever:
    def __init__(self, key, broker, udp_port, logger):
        """
        Initializes the Receiver class.

        Parameters:
        key (str): Key identifying the stream.
        broker (str): The Zenoh broker's address.
        udp_port (int): The UDP port for streaming.

        This method initializes the Receiver class and sets up the Zenoh connection and frames capturing.
        """
        self.key = key
        self.broker = broker
        self.logger = logger
        self.ffmpeg_process = (
            ffmpeg.input(
                "pipe:", format="rawvideo", pix_fmt="bgr24", s="1920x1080"
            )
            .output(
                f"rtsp://localhost:8554/{udp_port}",
                f="rtsp",
                vcodec="libx264",
                pix_fmt="yuv420p",
                rtsp_transport="tcp",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        # initiate logging
        zenoh.init_logger()
        self.logger.info("Opening session...")
        self.session = self.establish_zenoh_connection(max_retries=3)
        self.logger.info("Declaring Subscriber on '{}'...".format(self.key))

        self.capture_frames()

    def establish_zenoh_connection(self, max_retries=3):
        """
        Establishes a connection with the Zenoh broker.

        Parameters:
        max_retries (int): The maximum number of retries for connection.

        Returns:
        Zenoh session upon successful connection or exits if unsuccessful.

        This method attempts to establish a connection with the Zenoh broker, retrying a specific number of times.
        """
        retries = 0
        session = None

        while retries < max_retries:
            try:
                zenoh_config = zenoh.Config()
                zenoh_config.insert_json5(
                    zenoh.config.LISTEN_KEY, json.dumps([self.broker])
                )
                zenoh_config.insert_json5(
                    "scouting/multicast/enabled", "false"
                )
                session = zenoh.open(zenoh_config)
                self.logger.info(
                    "Zenoh broker connection established successfully."
                )
                return session
            except Exception:
                if retries < max_retries - 1:
                    self.logger.warning(
                        f"Retrying to get the Zenoh broker connection ({retries + 1}/{max_retries})"
                    )
                retries += 1
                if retries < max_retries:
                    time.sleep(5)

        self.logger.error(
            f"Zenoh broker connection cannot be established after {max_retries} retries. Exiting."
        )
        if session is not None:
            session.close()
        exit(1)

    def capture_frames(self):
        """
        Captures frames from the RTSP stream.

        This method captures frames from the specified RTSP stream and sets up the subscriber to listen for frames.
        """
        self.sub = self.session.declare_subscriber(
            self.key, self.listener, reliability=Reliability.RELIABLE()
        )

        while True:
            time.sleep(1)

    def listener(self, sample: Sample):
        """
        Listener method for receiving frames.

        Parameters:
        sample (Sample): Received sample from the Zenoh stream.

        This method processes and handles the received frames from the Zenoh stream.
        """

        # Get frame payload
        frame_bytes = np.frombuffer(sample.payload, dtype=np.uint8)
        height, width, channels = 1080, 1920, 3
        frame_bytes = frame_bytes.reshape((height, width, channels))
        self.ffmpeg_process.stdin.write(frame_bytes)


def start_stream(args):
    """
    Initiates the Receiver class for a stream.

    Parameters:
    args (tuple): Tuple containing arguments for the stream.

    This method starts the Receiver class for the specified RTSP stream with provided configurations.
    """
    key, broker, udp_port, logger = args
    Reciever(key, broker, udp_port, logger)


def process_streams(rtsp_streams, logger):
    """
    Processes multiple RTSP streams.

    Parameters:
    rtsp_streams (dict): Dictionary containing details of multiple RTSP streams.

    This method starts the Zenoh receiver process for each RTSP stream provided using a multiprocessing Pool.
    """
    pool_size = len(rtsp_streams)
    with Pool(processes=pool_size) as pool:
        args_list = []
        udp_port = 1234
        zenoh_port = 7447

        for stream_config in rtsp_streams:
            key = f"zenoh-pub-stream"
            broker = f"tcp/{rtsp_streams[stream_config]['broker']}:{str(zenoh_port)}"
            args_list.append((key, broker, udp_port, logger))
            udp_port += 1
            zenoh_port += 1

        pool.map(start_stream, args_list)
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

        if not isinstance(data, dict) or "robot_config" not in data:
            return False

        rtsp_streams = data["robot_config"]
        if not isinstance(rtsp_streams, dict):
            return False

        for stream_name, stream_data in rtsp_streams.items():
            if (
                not isinstance(stream_data, dict)
                or "url" not in stream_data
                or "broker" not in stream_data
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
        sys.exit(0)

    with open(config, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        rtsp_streams = config_data.get("robot_config", [])

    process_streams(rtsp_streams, logger)


if __name__ == "__main__":
    fire.Fire(main)
