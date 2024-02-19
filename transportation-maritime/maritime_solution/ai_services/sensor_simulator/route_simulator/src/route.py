# Created by Scalers AI for Dell Inc.

from asyncua import Client, Node
import asyncio

from datetime import datetime
import time
import pandas as pd
import yaml
import openmeteo_utils
import subprocess
import generate_report as report_gen
import os
import logging

class ContainerSimulator:
    def __init__(self, container_id, initial_temperature=10.0, initial_humidity=60.0):
        """
        Initialize a ContainerSimulator object with specified container ID, temperature, and humidity.

        Args:
        container_id (int or str): Identifier for the container.
        initial_temperature (float, optional): Initial temperature inside the container, in Celsius.
            Defaults to 10.0.
        initial_humidity (float, optional): Initial humidity inside the container, as a percentage.
            Defaults to 60.0.

        Returns:
        None
        """
        self.container_id = container_id
        self.temperature = initial_temperature
        self.humidity = initial_humidity

    def get_environment(self, row):
        """
        Retrieve environmental data (temperature and relative humidity) for a specific location and time.

        Args:
        row (list or tuple): A row of data containing information about the location and timestamp.
            The row should have the following structure: [timestamp, latitude, longitude].

        Returns:
        Tuple[float, float, float, float]: A tuple containing the latitude, longitude, temperature, and
        relative humidity for the specified location and time.
        """
        latitude = row[1]
        longitude = row[2]
        
        meteoutils = openmeteo_utils.OpenMeteo()
        (temp, humi) = meteoutils.get_metrics(latitude, longitude, datetime.strptime(str(row[0]), r"%Y-%m-%d %H:%M:%S"))
        Temperature = round(float(temp), 2)
        Relative_humidity = round(float(humi), 2)
        
        return latitude, longitude, Temperature, Relative_humidity

    def display_environment(self):
        """
        Display the environment data for the container.

        Returns:
        None
        """
        print(
            f"drycontainers,zone=StowageZone1,container={str(self.container_id)} Temperature={self.temperature:.2f},Humidity={self.humidity:.2f}\n"
        )

async def send_environment_data(client, container_id, temperature, humidity, latitude, longitude, container_nodes):
    """
    Send environment data to the OPC UA server for a specific container.

    Args:
    client: OPC UA client for communication with the server.
    container_id (int): The unique identifier of the container.
    temperature (float): The temperature value to be sent.
    humidity (float): The humidity value to be sent.
    latitude (float): The latitude value to be sent.
    longitude (float): The longitude value to be sent.
    container_nodes (dict): A dictionary containing OPC UA nodes for each container.

    Returns:
    None
    """
    namespace_idx = await client.get_namespace_index("location_simulators")
    device_node = f"/Objects/{namespace_idx}:location_{container_id}"
    var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Temperature"
    print(var_node)
    var = await client.nodes.root.get_child(var_node)
    await var.set_value(temperature)
    var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Humidity"
    print(var_node)
    var = await client.nodes.root.get_child(var_node)
    await var.set_value(humidity)
    var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Longitude"
    print(var_node)
    var = await client.nodes.root.get_child(var_node)
    await var.set_value(longitude)
    var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Latitude"
    print(var_node)
    var = await client.nodes.root.get_child(var_node)
    await var.set_value(latitude)
    
def overwrite_file(file_path, content, logger):
    """
    Overwrite the content of a file with the specified content.

    Args:
    file_path (str): The path to the file to be overwritten.
    content (str): The new content to be written to the file.

    Returns:
    None
    """
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

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
    """
    Load configuration, initialize container simulators, connect to OPC UA server, read CSV file, fetch weather data, update container environment, and generate reports.

    Returns:
    None
    """
    # Initialize logger
    logging.basicConfig(
        filename="app.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # Load configuration from YAML file
    with open("/src/config/simulator_config.yaml", "r") as ymlfile:
        data = yaml.safe_load(ymlfile)
        
    num_containers = 1
    
    # Create and initialize container simulators
    container_simulators = [ContainerSimulator(container_id=i) for i in range(1, num_containers+1)]

    # Connect to OPC UA server
    IPAddr = os.getenv("SERVER_IP", "localhost")
    url = f"opc.tcp://{IPAddr}:4840"  # Adjust the URL as needed
    client = await establish_opcua_connection(url, logger)

    # Example usage:
    file_path = "/src/voyage_text/route_status.txt"  
    report_status_file = "/src/voyage_text/report_status.txt"  
    completion_time_file = "/src/voyage_text/completion_status.txt"

    try:
        overwrite_file(file_path, "In Progress", logger)
        overwrite_file(report_status_file, "Pending", logger)
        # Register namespace, create 'Parameters' node, and store container nodes in a dictionary
        namespace_idx = await client.get_namespace_index("location_simulators")
        container_nodes = {i: client.get_node(f"ns={namespace_idx};i={i}") for i in range(num_containers)}
        
        # Reading the csv file
        df = pd.read_csv("/src/config/route.csv")
        
        # Reading total no of seconds
        total_seconds = int(data["simulator_templates"]["route_sim_time"])

        # Calculating interval time between each row
        interval_time = total_seconds/len(df)

        # Getting temperature and humidity values from api
        df["Latitude"] = df.apply(lambda x: round(x["Latitude"],2), axis=1)
        df["Longitude"] = df.apply(lambda x: round(x["Longitude"],2), axis=1)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        vals = df[["Timestamp", "Latitude", "Longitude"]].values.tolist()

        for i in range(len(vals)):
            for container_simulator in container_simulators:
                latitude, longitude, Temperature, Relative_humidity = container_simulator.get_environment(vals[i])
                
                await send_environment_data(client, container_simulator.container_id, Temperature, Relative_humidity, latitude, longitude, container_nodes)
            time.sleep(interval_time)
        overwrite_file(file_path, "Completed", logger)
        overwrite_file(report_status_file, "Generating...", logger)
        start = time.time()
        report_gen.generate()
        end = time.time()
        gen_time = f"Last Generated at:{str(datetime.now()).split('.')[0]}, Generation time: {end-start:.2f} s" 
        overwrite_file(report_status_file, "Generated", logger)
        overwrite_file(completion_time_file, gen_time, logger)
    finally:
        # Disconnect from the OPC UA server
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
