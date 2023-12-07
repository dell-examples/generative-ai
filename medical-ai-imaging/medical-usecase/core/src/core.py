# Created by Scalers AI for Dell Inc.

"""
DICOM Processor for Healthcare Applications
This script checks for changes in DICOM data from a PACS system, processes the data, and performs inference using an Inference Server.

It reads configurations from a YAML file, validates the PACS configuration, and initiates a continuous job to process DICOM data.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import inference_server
import schedule
import yaml
from dicom_processor import DicomProcessor


def validate_pacs_config(config):
    """
    Validates the PACS configuration from the loaded YAML file.

    Args:
    config (dict): Dictionary containing the configuration details.

    Returns:
    tuple or bool: If valid, returns a tuple of necessary parameters (URL, username, password, output_directory), else False.
    """
    required_keys = ["orthanc", "output_directory"]

    for key in required_keys:
        if key not in config or not config[key]:
            return False

    orthanc_info = config.get("orthanc", {})
    url = orthanc_info.get("url")
    username = orthanc_info.get("username")
    password = orthanc_info.get("password")
    output_directory = config.get("output_directory")

    if not all((url, username, password, output_directory)):
        return False

    return url, username, password, output_directory


def check_changes_job():
    """
    Check for changes in the DICOM data and process it.

    Returns:
    None
    """
    try:
        logger.info(f"Checking for changes at {datetime.now()}...")
        patient_ids = dicom_processor.get_patient_ids()
        for patient_id in patient_ids:
            dicom_processor.process_dicom(patient_id, output_directory)
        logger.info(f"Patient IDs: {patient_ids}")
    except Exception as e:
        logger.exception(f"An error occurred in the main job: {e}")


if __name__ == "__main__":
    # Initialize logger
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # Load configuration from YAML file
    with open("config.yaml", "r") as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            logger.error(exc)
            sys.exit(1)

    # Validate the configuration
    validated_values = validate_pacs_config(config)
    if validated_values:
        orthanc_url, username, password, output_directory = validated_values
        logger.info("Configuration parameters are valid.")
    else:
        logger.error("Please check the configuration file for errors.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_directory, exist_ok=True)
        logger.info(f"Output directory created: {output_directory}")
    except OSError as e:
        logger.error(f"Error creating directory: {e}")

    arg_values = config["inference"]
    args = argparse.Namespace(**arg_values)
    logger.info(f"Initializing with args {args}")
    infer_server = inference_server.Inference(args)
    infer_server.main()

    # Create an instance of the DicomProcessor class
    dicom_processor = DicomProcessor(
        orthanc_url, username, password, output_directory, infer_server, logger
    )

    interval_seconds = 5

    # Schedule the job to run at the defined interval
    schedule.every(interval_seconds).seconds.do(check_changes_job)

    # Run the job continuously
    while True:
        schedule.run_pending()
        time.sleep(1)
