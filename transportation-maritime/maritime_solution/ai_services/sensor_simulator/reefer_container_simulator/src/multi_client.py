# Created by Scalers AI for Dell Inc.

from opcua import Client as SyncClient
import random
import time
import pandas as pd
import yaml
import os
import logging

class ContainerSimulator:
    def __init__(self, container_id, initial_temperature=4.0):
        """
        Initialize a ContainerSimulator object with specified container ID and initial temperature.

        Parameters:
        - container_id (str): The unique identifier for the container.
        - initial_temperature (float): The initial temperature inside the container,
          defaults to 4.0 degrees Celsius if not provided.
        """
        self.container_id = container_id
        self.temperature = initial_temperature

    def update_environment(self, temperature):
        """
        Update the environment temperature of the container.

        Parameters:
        - temperature (float): The new temperature value to be set for the container environment.

        Returns:
        None
        """

        # Update environment based on provided temperature values
        self.temperature = max(0.0, min(60.0, temperature + random.uniform(-0.5, 0.5)))

    def get_environment(self):
        """
        Get the current temperature of the container environment.

        Returns:
        - temperature (float): The current temperature of the container environment.
        """
        return self.temperature

    def display_environment(self):
        """
        Display the current temperature of the container environment.

        Prints:
        - A formatted string representing the container's environment with temperature value.
        """
        print(
            f"reefer,zone=StowageZone2,container={str(self.container_id)} Temperature={self.temperature:.2f}\n"
        )

def send_environment_data(client, container_id, temperature, container_nodes):
    """
    Send temperature data to a client representing a container node in an IoT system.

    Parameters:
    - client: The client representing the connection to the container node.
    - container_id (str): The unique identifier of the container.
    - temperature (float): The temperature value to be sent.
    - container_nodes (dict): A dictionary mapping container IDs to their corresponding nodes.

    Returns:
    None
    """
    try:
        node = container_nodes.get(container_id)
        if node:
            temperature_variable = node.get_child(["3:Temperature"])
            temperature_variable.set_value(temperature)
        else:
            pass
    except Exception as e:
        pass

def establish_opcua_connection(url, logger, max_retries=12):
    """
    Establish a connection to OPCUA Server.

    Parameters:
    - max_retries (int): The maximum number of retries to establish the connection, defaults to 3.

    Returns:
    None
    """
    retries = 0
    while retries < max_retries:
        try:
            client = SyncClient(url)
            client.connect()
            return client
        except Exception:
            if retries < max_retries - 1:
                logger.error(f"Retrying to get the OPCUA connection ({retries + 1}/{max_retries})")
            retries += 1
            if retries < max_retries:
                time.sleep(5)

def main():
    # Load configuration from YAML file
    with open("/app/config/simulator_config.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        
    num_containers = config["cargo_info"]["total_containers"] - config["cargo_info"]["reefer_containers"]
    csv_path = config["simulator_templates"]["reefer_template"]
    # Load temperature data from CSV file
    data = pd.read_csv(csv_path, parse_dates=['Timestamp'])
    
    # Create and initialize container simulators
    container_simulators = [ContainerSimulator(container_id=i) for i in range(num_containers)]

    # Initialize logger
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # Connect to OPC UA server
    IPAddr = os.getenv("SERVER_IP", "localhost")
    url = f"opc.tcp://{IPAddr}:4850"  # Adjust the URL as needed
    client = establish_opcua_connection(url, logger)

    try:
        # Register namespace, create 'Parameters' node, and store container nodes in a dictionary
        namespace_idx = client.get_namespace_index("REEFER_SERVER")
        container_nodes = {i: client.get_node(f"ns={namespace_idx};i={i}") for i in range(num_containers)}

        # Iterate through each row in the CSV data
        for i in range(config["voyage_details"]["total_voyage_duration_days"]):
            for _, row in data.iterrows():

                # Update environment for each container based on provided temperature values
                for container_simulator in container_simulators:
                    container_simulator.update_environment(row['Temperature'])

                # Display container environments
                for container_simulator in container_simulators:
                    container_simulator.display_environment()

                # Send simulated environment data to the OPC UA server for each container
                for container_simulator in container_simulators:
                    temperature = container_simulator.get_environment()
                    send_environment_data(client, container_simulator.container_id, temperature, container_nodes)

                # Sleep for a short duration to simulate the journey time
                time.sleep(5)

    finally:
        # Disconnect from the OPC UA server
        client.disconnect()

if __name__ == "__main__":
    main()
