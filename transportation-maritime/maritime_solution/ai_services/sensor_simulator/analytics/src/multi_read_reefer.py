# Created by Scalers AI for Dell Inc.

from asyncua import Client, Node
import asyncio
import logging
import time
import pandas as pd
import pickle
import os
import yaml

async def read_environment_data(client, container_nodes, model, logger):
    """
    Read environment data from container nodes and predict power consumption using a machine learning model.

    Parameters:
    - client: The client representing the connection to the container nodes.
    - container_nodes (dict): A dictionary mapping container IDs to their corresponding nodes.
    - model: The machine learning model used for predicting power consumption.

    Returns:
    None
    """

    for container_id, node in container_nodes.items():
        try:
            
            external_temperature = await client.get_namespace_index("location_simulators")
            device_node = f"/Objects/{external_temperature}:location_1"
            var_node = f"{device_node}/{int(external_temperature)}:TH_sensor/{int(external_temperature)}:Temperature"
            var = await client.nodes.root.get_child(var_node)
            temperature_value = await var.get_value()
            
            dev_data = "/app/config/simulator_config.yaml"
            with open(dev_data, "r") as c_file:
                try:
                    config = yaml.safe_load(c_file)
                except yaml.YAMLError as err:
                    err_msg = f"Error while parsing config file: {err}."
                    logger.error(err_msg)
            set_temperature_value = int(config["cargo_info"]["reefer_set_temp"])
            val = pd.DataFrame({'Temperature': [temperature_value], 'Set Temperature': [set_temperature_value]})
            predicted_value = model.predict(val)
            print(
                f"powerconsumption,zone=StowageZone1,container=EMCU{str(container_id)} Power={predicted_value[0]:.2f}\n"
            )    
        except Exception as e:
            pass

    
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
    num_containers = 1

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
        # Register namespace and get namespace index
        namespace_idx = await client.get_namespace_index("location_simulators")

        # Get nodes for each container
        container_nodes = {i: client.get_node(f"ns={namespace_idx};i={i}") for i in range(num_containers)}

        with open("power_estimation_model.pkl", "rb") as file:
            model = pickle.load(file)

        # Read and display environment data from the OPC UA server
        while True:
            await read_environment_data(client, container_nodes, model, logger)
            time.sleep(0.1)  # Adjust the sleep duration as needed

    finally:
        # Disconnect from the OPC UA server
        client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
