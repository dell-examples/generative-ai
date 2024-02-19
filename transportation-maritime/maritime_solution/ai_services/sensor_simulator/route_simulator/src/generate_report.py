# Created by Scalers AI for Dell Inc.

import yaml
import sys
import requests
import os
import shutil
import datetime
from context_sweat import ContextSweat
from context_power import ContextPower
from context_worker import ContextWorker
import time
import logging

def get_power_consumption(IPAddr, logger):
    """
    Get power consumption insights of the voyage.

    Returns:
    str: A string containing the power consumption insights obtained from the service.
    """

    response = requests.get(f'http://{IPAddr}:8000/qa/predict?query="Summarize this power consumption system for this voyage in only one paragraph, give me the maximum, minimum and exceptions if any"')

    text = response.text
    text = text.replace("<code>", "")
    text = text.replace("</code>", "")
    text = text.replace("<pre>", "\n")
    text = text.replace("</pre>", "\n")
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    text = text.replace('"',"")

    logger.info(text)
    return text

def get_sweat_info(IPAddr, logger):
    """
    Get sweat alerts information for the voyage.

    This function sends a GET request to a QA (Question Answering) service endpoint
    to obtain information summarizing the sweat alerts for the voyage. The endpoint
    expects a query string formatted with a specific question related to summarizing
    sweat alerts.

    Returns:
    str: A string containing the sweat alerts information obtained from the service.
    """
    response = requests.get(f'http://{IPAddr}:8000/qa/predict?query="Summarize the sweat alerts for this voyage in only one paragraph."')

    text = response.text
    text = text.replace("<code>", "")
    text = text.replace("</code>", "")
    text = text.replace("<pre>", "\n")
    text = text.replace("</pre>", "\n")
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    text = text.replace('"',"")

    text = "\n".join(line for line in text.splitlines() if line.strip())

    with open("/src/voyage_text/events/sweat.txt", "r") as table_txt:
        val = "".join([line for line in table_txt if line.strip()][2:])
        text = text + "\n\n**Events Summary:**\n\n" + val
    
    logger.info(text)
    return text

def get_incidents(IPAddr, logger):
    """
    Get worker zone violation incidents summary for the voyage.

    Returns:
    str: A string containing the worker zone violation incidents summary obtained
    from the service.
    """
    response = requests.get(f'http://{IPAddr}:8000/qa/predict?query="Summarize the worker zone violations for this voyage in only one paragraph."')

    text = response.text
    text = text.replace("<code>", "")
    text = text.replace("</code>", "")
    text = text.replace("<pre>", "\n")
    text = text.replace("</pre>", "\n")
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    text = text.replace('"',"")
    
    logger.info(text)
    return text

def get_conclusion(final_str, IPAddr, logger):
    """
    Get the conclusion of the voyage.

    Returns:
    str: A string containing the summary conclusion obtained from the service.
    """

    # Set the folder name
    folder_name = '/src/voyage_text/events_conclusion/'

    # Set the file name
    file_name = 'conclusion.txt'

    # Create the folder if it doesn't exist, or delete and recreate if it does
    if not os.path.exists(folder_name):
        # If the folder dosen't exist, recreate
        os.makedirs(folder_name)

    # Create and write to the file
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, 'w') as file:
        file.write(final_str)

    query = "Summarize this voyage and give me the conclusion in only one paragraph."
    response = requests.get(f'http://{IPAddr}:8000/qa/conclusion?query={query}')

    text = response.text
    text = text.replace("<code>", "")
    text = text.replace("</code>", "")
    text = text.replace("<pre>", "\n")
    text = text.replace("</pre>", "\n")
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")

    logger.info(text)
    return f"""

{text}

    """

def generate_report(template, config, logger):
    """
    Generate a report based on the template and the config file.

    Returns:
    str: A string containing the generated report.
    """
    start = time.time()
    container_types_string = f'{config["cargo_info"]["total_containers"]} x {config["cargo_info"]["container_type"]}'

    IPAddr = os.getenv("SERVER_IP", "localhost")

    final_str = template.format(
        ip_address=IPAddr, 
        voyage_number=config["voyage_details"]["voyage_number"],
        vessel_name = config["voyage_details"]["vessel_name"],
        captain_name = config["voyage_details"]["captain"],
        departure_port = config["voyage_details"]["departure_port"],
        arrival_port = config["voyage_details"]["arrival_port"],
        departure_date = config["voyage_details"]["departure_date"],
        arrival_date = config["voyage_details"]["arrival_date_est"],
        total_duration = config["voyage_details"]["total_voyage_duration_days"],
        total_containers = config["cargo_info"]["total_containers"],
        container_types = container_types_string,
        total_dry_containers = config["cargo_info"]["total_containers"] - config["cargo_info"]["reefer_containers"],
        total_reefer_containers = config["cargo_info"]["reefer_containers"],
        power_consumption = get_power_consumption(IPAddr, logger),
        sweat_info = get_sweat_info(IPAddr, logger),
        incidents = get_incidents(IPAddr, logger),
        generation_time = str(datetime.datetime.now()),
        )

    final_str = final_str + "\n" + str(get_conclusion(final_str, IPAddr, logger))
    end = time.time()

    logger.info(end-start)

    return final_str

def generate():
    """
    Generate the final voyage report.
    
    Returns:
    None
    """
    start = time.time()
    ContextSweat()
    ContextPower()
    ContextWorker()
    end = time.time()

    # Initialize logger
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    logger.info(end-start)
    template = ""
    with open("/src/voyage_text/template.md", "r") as f:
        try:
            template = f.read()
        except Exception as err:
            err_msg = f"Error while reading template file: {err}."
            logger.error(err_msg)
            sys.exit(1)

    final_report_md = ""

    dev_data = "/src/config/simulator_config.yaml"
    with open(dev_data, "r") as c_file:
        try:
            config = yaml.safe_load(c_file)
        except yaml.YAMLError as err:
            err_msg = f"Error while parsing config file: {err}."
            logger.error(err_msg)
            sys.exit(1)

        final_report_md = generate_report(template, config, logger)

    with open("/src/voyage_text/final_report.md", "w") as f:
        f.write(final_report_md)

if __name__ == "__main__":

    generate()
