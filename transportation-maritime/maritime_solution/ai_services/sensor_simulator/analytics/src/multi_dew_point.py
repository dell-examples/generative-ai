# Created by Scalers AI for Dell Inc.

from asyncua import Client, Node
import asyncio
import logging
import time
import yaml
import os

async def read_environment_data(client, container_nodes, num_containers, logger):
    """
    Read environment data from container nodes and compare it with external temperature for dew point calculation.

    Parameters:
    - client: The client representing the connection to the OPC UA server.
    - container_nodes (dict): A dictionary mapping container IDs to their corresponding nodes.

    Returns:
    None
    """
    try:
        external_temperature = await client.get_namespace_index("location_simulators")
        device_node = f"/Objects/{external_temperature}:location_1"
        var_node = f"{device_node}/{int(external_temperature)}:TH_sensor/{int(external_temperature)}:Temperature"
        var = await client.nodes.root.get_child(var_node)
        temperature_ext_value = await var.get_value()
            
        alert_count = 0
        # for container_id, node in container_nodes.items():
        for container_id in range(1, num_containers+1):
            
            namespace_idx = await client.get_namespace_index("container_simulators")
            device_node = f"/Objects/{namespace_idx}:dry_container_{container_id}"
            var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Temperature"
            var = await client.nodes.root.get_child(var_node)
            temperature_value = await var.get_value()
            var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Humidity"
            var = await client.nodes.root.get_child(var_node)
            humidity_value = await var.get_value()
                
            dew_pt_temp = temperature_value - ((100-humidity_value)/5)


            if dew_pt_temp > temperature_ext_value:
                print(
                    f'dewpoint,zone=StowageZone1,container=EMCU{str(container_id)},AlertTag=True Dewpoint={dew_pt_temp:.2f},ContainerID="EMCU{str(container_id)}",Alert=True\n'
                )
                alert_count += 1
            else:
                print(
                    f'dewpoint,zone=StowageZone1,container=EMCU{str(container_id)},AlertTag=True Dewpoint={dew_pt_temp:.2f},ContainerID="EMCU{str(container_id)}",Alert=False\n'
                )

        print(
            f'alert,zone=StowageZone1 AlertCount={alert_count}\n'
        )
    except Exception as e:
        logger.error(f"Error reading environment data: {e}")

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
    with open("/app/config/simulator_config.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        
    num_containers = config["cargo_info"]["total_containers"] - config["cargo_info"]["reefer_containers"]

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
        namespace_idx = await client.get_namespace_index("container_simulators")

        # Get nodes for each container
        container_nodes = {i: client.get_node(f"ns={namespace_idx};i={i}") for i in range(num_containers)}

        # Read and display environment data from the OPC UA server
        while True:
            await read_environment_data(client, container_nodes, num_containers, logger)
            time.sleep(1)  # Adjust the sleep duration as needed

    finally:
        # Disconnect from the OPC UA server
        client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
