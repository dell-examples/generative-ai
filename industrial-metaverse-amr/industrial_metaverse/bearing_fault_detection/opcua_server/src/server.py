# Created by Scalers AI for Dell Inc.

import asyncio
import os

import yaml
from asyncua import Server


async def main():
    # Load the configuration from the YAML file
    with open("/config/simulator_config.yml", "r") as config:
        yaml_config = yaml.safe_load(config)

    server = Server()
    await server.init()

    opcua_endpoint_url = os.getenv(
        "OPCUA_SERVER", "opc.tcp://opcua-server:4840"
    )
    server.set_endpoint(opcua_endpoint_url)

    # Register OPC UA namespace(s)
    namespaces = yaml_config.get("namespaces", {})
    for index, (namespace_name, namespace_config) in enumerate(
        namespaces.items(), start=1
    ):
        # create the namespace
        namespace_idx = await server.register_namespace(namespace_name)

        # get all the devices
        devices = namespace_config.get("devices", {})
        for _, (device_name, device_config) in enumerate(
            devices.items(), start=1
        ):
            # replica handling
            device_replicas = device_config.get("replicas", 1)
            for index in range(1, device_replicas + 1):
                # add a device object
                replica_device_name = f"{device_name}_{index}"
                device_node = await server.nodes.objects.add_object(
                    namespace_idx, replica_device_name
                )
                # iterate over all sensor nodes
                sensor_nodes = device_config.get("sensor_nodes", [])
                for index, (sensor_name, sensor_config) in enumerate(
                    sensor_nodes.items(), start=1
                ):
                    # add each sensor nodes as objects under device
                    sensor_node = await device_node.add_object(
                        namespace_idx, sensor_name
                    )
                    # add variables under each sensors
                    for variable in sensor_config["variables"]:
                        sensor_var = await sensor_node.add_variable(
                            namespace_idx, variable, 0.0
                        )
                        await sensor_var.set_writable()

    print("Starting OPC UA Server....")
    # Start the OPC UA server
    async with server:
        while True:
            # Keep the server script running indefinitely
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
