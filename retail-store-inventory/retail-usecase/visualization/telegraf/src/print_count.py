# Created by Scalers AI for Dell Inc.

import json
import logging
import os
import sys
import time

import fire
import yaml
import zenoh
from zenoh import Reliability, Sample


class Reciever:
    def __init__(self, key, broker):
        self.key = key
        self.broker = broker
        # initiate logging
        zenoh.init_logger()
        self.session = self.establish_zenoh_connection(max_retries=3)
        self.combined_current_dist = {}
        self.combined_total_dist = {}
        self.last_frame_info = {}
        self.get_metadata()

    def establish_zenoh_connection(self, max_retries=3):
        retries = 0
        session = None

        while retries < max_retries:
            try:
                zenoh_config = zenoh.Config()
                zenoh_config.insert_json5(
                    zenoh.config.CONNECT_KEY, json.dumps([self.broker])
                )
                session = zenoh.open(zenoh_config)
                logging.info(
                    "Zenoh broker connection established successfully."
                )
                return session
            except Exception:
                if retries < max_retries - 1:
                    logging.warning(
                        f"Retrying to get the Zenoh broker connection ({retries + 1}/{max_retries})"
                    )
                retries += 1
                if retries < max_retries:
                    time.sleep(5)

        logging.error(
            f"Zenoh broker connection cannot be established after {max_retries} retries. Exiting."
        )
        if session is not None:
            session.close()
        exit(1)

    def get_metadata(self):
        self.sub = self.session.declare_subscriber(
            self.key, self.listener, reliability=Reliability.RELIABLE()
        )

        while True:
            time.sleep(1)

    def listener(self, sample: Sample):
        key = sample.key_expr

        # Check if this is the first frame for this stream
        if key not in self.last_frame_info:
            self.last_frame_info[key] = {
                "timestamp": time.time(),
                "frame_counter": 0,
            }
        else:
            current_time = time.time()
            time_elapsed = (
                current_time - self.last_frame_info[key]["timestamp"]
            )
            self.last_frame_info[key]["frame_counter"] += 1
            if time_elapsed >= 1:
                self.last_frame_info[key]["frame_counter"] / time_elapsed
                self.last_frame_info[key] = {
                    "timestamp": current_time,
                    "frame_counter": 0,
                }

                # Get frame payload
                frame_bytes = sample.payload
                # Convert bytes to string
                json_string = frame_bytes.decode("utf-8")
                # Convert string to JSON object
                json_object = json.loads(json_string)
                formatted_data = {}
                for key, value in json_object.items():
                    for sub_key, sub_value in value.items():
                        formatted_data[f"{key}_{sub_key}"] = sub_value

                for key, value in formatted_data.items():
                    for sub_key, sub_value in value.items():
                        if sub_key == "current":
                            self.combined_current_dist[key] = int(sub_value)
                        elif sub_key == "total":
                            self.combined_total_dist[key] = int(sub_value)

                lines = []
                for tag, fields in formatted_data.items():
                    field_strings = []
                    tag_string = f"zone={tag}"
                    measurement_name = "retail"
                    for field_name, field_value in fields.items():
                        field_strings.append(f"{field_name}={field_value}")

                    field_string = ",".join(field_strings)
                    lines.append(
                        f"{measurement_name},{tag_string} {field_string}"
                    )
                    print(f"{measurement_name},{tag_string} {field_string}\n")
                print(
                    f"combined,zone=combined total={sum(self.combined_total_dist.values())},current={sum(self.combined_current_dist.values())}\n"
                )


def start_receiver(key, broker):
    Reciever(key, broker)


def process_rtsp_streams(rtsp_streams):
    zenoh_port = 7447
    for stream_config in rtsp_streams:
        key = "metadata"
        broker = (
            f"tcp/{rtsp_streams[stream_config]['broker']}:{str(zenoh_port)}"
        )
        start_receiver(key, broker)
        zenoh_port += 1


def validate_streams_yaml(yaml_file):
    """
    Validates the format of a YAML configuration file.

    Parameters:
    yaml_file (str): File path of the YAML configuration.

    Returns:
    bool: True if the YAML format is valid, otherwise False.
    """
    if not os.path.isfile(yaml_file):
        logging.error(f"The file {yaml_file} does not exist.")
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
    config: str = "config.yaml",
):
    # check if yaml file exists and is in correct format
    if validate_streams_yaml(config):
        logging.info(f"The format of {config} is valid.")
    else:
        logging.error(f"The format of {config} is not valid.")
        sys.exit(0)

    with open(config, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        rtsp_streams = config_data.get("rtsp_streams", [])

    process_rtsp_streams(rtsp_streams)


if __name__ == "__main__":
    fire.Fire(main)
