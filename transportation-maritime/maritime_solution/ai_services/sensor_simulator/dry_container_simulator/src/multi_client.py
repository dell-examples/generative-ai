# Created by Scalers AI for Dell Inc.

import random
import time
import pandas as pd
import yaml
import os
import logging
import asyncio
from asyncua import Client, Node

class ContainerSimulator:
    def __init__(self, container_id, initial_temperature=10.0, initial_humidity=60.0):
        """
        Initialize a ContainerSimulator object with specified container ID,
        initial temperature, and initial humidity.

        Parameters:
        - container_id (str): The unique identifier for the container.
        - initial_temperature (float): The initial temperature inside the container,
          defaults to 10.0 degrees Celsius if not provided.
        - initial_humidity (float): The initial humidity level inside the container
          as a percentage, defaults to 60.0 if not provided.
        """
        self.container_id = container_id
        self.temperature = initial_temperature
        self.humidity = initial_humidity

    def update_environment(self, temperature, humidity, is_normal):
        """
        Update the environment parameters of the container based on the provided temperature,
        humidity, and a flag indicating whether the update is for normal conditions.

        Parameters:
        - temperature (float): The new temperature value to be set for the container environment.
        - humidity (float): The new humidity value to be set for the container environment.
        - is_normal (bool): A boolean flag indicating whether the update is for normal conditions.
        If True, the temperature and humidity will be adjusted within a normal range. If False,
        the temperature and humidity may be adjusted to simulate abnormal conditions.

        Returns:
        None
        """
        if is_normal:
            # Update environment based on provided temperature and humidity values
            self.temperature = max(0.0, min(60.0, temperature + random.uniform(-2, 2)))
            self.humidity = max(0.0, min(100.0, humidity + random.uniform(-2, 2)))
        else:
            # Update environment based on provided temperature and humidity values
            self.temperature = max(40.0, min(60.0, temperature + random.uniform(-2, 2)))
            self.humidity = 94.54 + random.randint(0, 5)

    def get_environment(self):
        """
        Get the current temperature and humidity of the container environment.

        Returns:
        - temperature (float): The current temperature of the container environment.
        - humidity (float): The current humidity of the container environment.
        """
        return self.temperature, self.humidity

    def display_environment(self):
        """
        Display the current temperature and humidity of the container environment in a formatted string.

        Prints:
        - A formatted string representing the container's environment with temperature and humidity values.
        """
        print(
            f"drycontainers,zone=StowageZone1,container={str(self.container_id)} Temperature={self.temperature:.2f},Humidity={self.humidity:.2f}\n"
        )

async def send_environment_data(client, container_id, temperature, humidity, container_nodes):
    """
    Send temperature and humidity data to a client representing a container node in an IoT system.

    Parameters:
    - client: The client representing the connection to the container node.
    - container_id (str): The unique identifier of the container.
    - temperature (float): The temperature value to be sent.
    - humidity (float): The humidity value to be sent.
    - container_nodes (dict): A dictionary mapping container IDs to their corresponding nodes.

    Returns:
    None
    """
    namespace_idx = await client.get_namespace_index("container_simulators")
    device_node = f"/Objects/{namespace_idx}:dry_container_{container_id}"
    var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Temperature"
    var = await client.nodes.root.get_child(var_node)
    await var.set_value(round(temperature, 2))
    var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Humidity"
    var = await client.nodes.root.get_child(var_node)
    await var.set_value(round(humidity, 2))    
    
async def establish_opcua_connection(url, logger, max_retries=12):
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
            client = Client(url=url)
            await client.connect()
            return client
        except Exception:
            if retries < max_retries - 1:
                logger.error(f"Retrying to get the OPCUA connection ({retries + 1}/{max_retries})")
            retries += 1
            if retries < max_retries:
                time.sleep(5)

async def main():
    # Load configuration from YAML file
    with open("/app/config/simulator_config.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        
    num_containers = config["cargo_info"]["total_containers"] - config["cargo_info"]["reefer_containers"]
    csv_path = config["simulator_templates"]["dry_container_template"]
    # Load temperature and humidity data from CSV file
    data = pd.read_csv(csv_path, parse_dates=['Timestamp'])
    
    # Create and initialize container simulators
    container_simulators = [ContainerSimulator(container_id=i) for i in range(1, num_containers+1)]

    # Initialize logger
    logging.basicConfig(
        filename="app.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # Connect to OPC UA server
    IPAddr = os.getenv("SERVER_IP", "localhost")
    url = f"opc.tcp://{IPAddr}:4840"  # Adjust the URL as needed
    client = await establish_opcua_connection(url, logger)

    try:
        # Register namespace, create 'Parameters' node, and store container nodes in a dictionary
        namespace_idx = await client.get_namespace_index("container_simulators")

        
        container_nodes = {i: client.get_node(f"ns={namespace_idx};i={i}") for i in range(num_containers)}

        # Calculate delay between each data publish
        delay = config["simulator_templates"]["route_sim_time"]/24

        start_time = time.time()
        indx = 0

        container_exceptions = []
        for size in [30,50,10]:
            inner_list = [random.randint(1, num_containers) for _ in range(size)]
            container_exceptions.append(inner_list)

        exception_times = []
        for percentage in [0.05, 0.35, 0.80]:
            time_value = int(config["simulator_templates"]["route_sim_time"] * percentage)
            exception_times.append([time_value, time_value + 10])

        # Iterate through each row in the CSV data
        # for i in range(config["voyage_details"]["total_voyage_duration_days"]):
        # while True:
        for j, row in data.iterrows():
            timestamp = row['Timestamp']         

            if indx >= len(exception_times):
                indx = len(exception_times) - 1
            if (time.time() - start_time > exception_times[indx][0]) and (time.time() - start_time < exception_times[indx][1]):
            # Update environment for each container based on provided temperature and humidity values
                for idx, container_simulator in enumerate(container_simulators):
                        if idx in container_exceptions[indx]:
                            container_simulator.update_environment(row['Temperature'], row['Humidity'], False)
                        else:
                            container_simulator.update_environment(row['Temperature'],row['Humidity'], True)
            else:
                for idx, container_simulator in enumerate(container_simulators):
                    container_simulator.update_environment(row['Temperature'],row['Humidity'], True)
            if time.time() - start_time >= exception_times[indx][1]:
                indx += 1
                

            # Display container environments
            for container_simulator in container_simulators:
                container_simulator.display_environment()

            # Send simulated environment data to the OPC UA server for each container
            for container_simulator in container_simulators:
                temperature, humidity = container_simulator.get_environment()
                await send_environment_data(client, container_simulator.container_id, temperature, humidity, container_nodes)

            # Sleep for a short duration to simulate the journey time
            time.sleep(delay)

    finally:
        # Disconnect from the OPC UA server
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
