import asyncio
import os
import csv
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from asyncua import Client
from fastapi.staticfiles import StaticFiles

# Create a FastAPI application instance
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# Number of namespaces
num_containers = 1

# Connect to OPC UA server
IPAddr = os.getenv("AI_SERVICES_SERVER_IP", "localhost")
url = f"opc.tcp://{IPAddr}:4840"  # Adjust the URL as needed

# Initialize logger
logging.basicConfig(
    filename="app.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


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
                await asyncio.sleep(5)


# Register namespace and get namespace index
async def init_opcua_client():
    client = await establish_opcua_connection(url, logger)
    namespace_idx = await client.get_namespace_index("location_simulators")
    container_nodes = {i: client.get_node(f"ns={namespace_idx};i={i}") for i in range(num_containers)}
    return client, container_nodes


async def read_environment_data(client, container_nodes, logger):
    """
    Read environment data (longitude and latitude) for each container node.

    Args:
        client: The OPC UA client.
        container_nodes (dict): A dictionary containing container nodes, where keys are container IDs and values are corresponding OPC UA nodes.

    Returns:
        tuple: A tuple containing latitude and longitude values for each container node.
    """
    latitude_values = []
    longitude_values = []
    for container_id, node in container_nodes.items():
        namespace_idx = await client.get_namespace_index("location_simulators")
        device_node = f"/Objects/{namespace_idx}:location_1"
        var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Longitude"
        var = await client.nodes.root.get_child(var_node)
        longitude_value = await var.get_value()
        var_node = f"{device_node}/{int(namespace_idx)}:TH_sensor/{int(namespace_idx)}:Latitude"
        var = await client.nodes.root.get_child(var_node)
        latitude_value = await var.get_value()
        latitude_values.append(latitude_value)
        longitude_values.append(longitude_value)
    return latitude_values, longitude_values


@app.get("/")
async def index():
    """
    Render the index.html template.

    Returns:
        FileResponse: Returns the index.html file from the static directory.
    """
    try:
        return FileResponse("templates/index.html")
    except Exception as e:
        raise HTTPException(status_code=404, detail="Index file not found")


@app.get('/get_ship_data')
async def get_ship_data():
    """
    Retrieve ship data from a CSV file and return it as JSON.

    Returns:
        str: JSON representation of ship data containing latitude and longitude.
    """
    json_data = []
    with open("/src/config/route.csv", 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            latitude = float(row['Latitude'])
            longitude = float(row['Longitude'])
            json_data.append({'Latitude': round(latitude, 2), 'Longitude': round(longitude, 2)})
    return json_data


@app.get('/current_location')
async def current_location():
    """
    Retrieve the current latitude and longitude of a container and return it as JSON.

    Returns:
        dict: A JSON object containing the latitude and longitude of the container.
    """
    try:
        client, container_nodes = await init_opcua_client()
        latitude_values, longitude_values = await read_environment_data(client, container_nodes, logger)
        print({"Latitude": latitude_values, "Longitude": longitude_values})
        return {"Latitude": latitude_values, "Longitude": longitude_values}
    except Exception as e:
        logger.error(f"Error retrieving current location: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
