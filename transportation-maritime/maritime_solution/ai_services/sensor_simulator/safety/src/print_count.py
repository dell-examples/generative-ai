# Created by Scalers AI for Dell Inc.

import json
import time
from multiprocessing import Pool

import fire
import yaml
import zenoh
from zenoh import Reliability, Sample


class Reciever:
    def __init__(self, key, broker, udp_port):
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
        # initiate logging
        zenoh.init_logger()
        self.session = self.establish_zenoh_connection(max_retries=3)
        self.time_start = time.time()
        self.line_text = ""
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
                
                return session
            except Exception:
                if retries < max_retries - 1:
                    print(
                        f"Retrying to get the Zenoh broker connection ({retries + 1}/{max_retries})"
                    )
                retries += 1
                if retries < max_retries:
                    time.sleep(5)

        print(
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
        key = sample.key_expr

        # Get frame payload
        key = sample.key_expr
        if (time.time() - self.time_start > 5): 
            # Get frame payload
            frame_bytes = sample.payload
            # Convert bytes to string
            json_string = frame_bytes.decode("utf-8")
            # Convert string to JSON object
            json_object = json.loads(json_string)
            self.line_text = f"combined,zone={json_object['Stream ID']} total={json_object['Total Violations']},current={json_object['Total People']}\n"
            self.time_start = time.time()
        print(self.line_text)
def start_stream(args):
    """
    Initiates the Receiver class for a stream.

    Parameters:
    args (tuple): Tuple containing arguments for the stream.

    This method starts the Receiver class for the specified RTSP stream with provided configurations.
    """
    key, broker, udp_port = args
    Reciever(key, broker, udp_port)


def process_streams(rtsp_streams):
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
            key = f"metadata"
            broker = f"tcp/{rtsp_streams[stream_config]['broker']}:{str(zenoh_port)}"
            args_list.append((key, broker, udp_port))
            udp_port += 1
            zenoh_port += 1
        pool.map(start_stream, args_list)
        pool.close()
        pool.join()


def validate_streams_yaml(yaml_file):
    """
    Validates the format of a YAML configuration file.

    Parameters:
    yaml_file (str): File path of the YAML configuration.

    Returns:
    bool: True if the YAML format is valid, otherwise False.
    """
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
            ):
                return False

        return True

    except yaml.YAMLError:
        return False


def main(
    config: str = "/app/config/simulator_config.yaml",
):

    with open(config, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        rtsp_streams = config_data.get("rtsp_streams", [])

    process_streams(rtsp_streams)


if __name__ == "__main__":
    fire.Fire(main)
