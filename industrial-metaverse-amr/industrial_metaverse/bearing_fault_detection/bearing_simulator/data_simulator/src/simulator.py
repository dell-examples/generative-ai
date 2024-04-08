# Created by Scalers AI for Dell Inc.

import asyncio
import os
import random
import time

import pandas as pd
import yaml
from asyncua import Client, Node


class OPCUASimulator:
    def __init__(self, config_file_path: str) -> None:
        """
        Initialize the OPCUASimulator with the configuration file path.
        :param config_file_path: Path to the YAML configuration file.
        """
        self.config_file_path = config_file_path
        self.opcua_endpoint_url = None
        self.simulator_namespaces = {}
        self.max_retries = 3
        self.retry_interval = 5

    def load_configuration(self) -> None:
        """
        Load the configuration from the YAML file.
        """
        with open(self.config_file_path, "r") as config:
            yaml_config = yaml.safe_load(config)

        self.opcua_endpoint_url = os.getenv(
            "OPCUA_SERVER", "opc.tcp://localhost:4840"
        )
        self.simulator_namespaces = yaml_config.get("namespaces", {})

    async def push_to_opcua(self, value: float, node: Node) -> None:
        """Write the value to the OPC UA server.

        :param value: The value to be written.
        :param node: The OPC UA node to write the value to.
        """
        node.set_value(value)

    async def publish_data_for_device(
        self, client, namespace_index, device, device_config, interval
    ):
        base_path = device_config.get("base_path", "./simulator_data")
        simulator_config = device_config.get("simulator_files")
        sensor_nodes = device_config.get("sensor_nodes", [])

        # read sensor nodes details of the device
        all_variables = []
        for sensor_name, sensor_config in sensor_nodes.items():
            for variable in sensor_config["variables"]:
                var_node = f"{int(namespace_index)}:{sensor_name}/{int(namespace_index)}:{variable}"
                all_variables.append(var_node)

        for _, sim_config in simulator_config.items():
            file_name = sim_config.get("file_name")
            row_count = sim_config.get("row_count")

            if file_name and row_count:
                file_path = os.path.join(base_path, file_name)
                df = pd.read_csv(file_path)

                # replicas handling
                device_replicas = device_config.get("replicas", 1)
                rate_changes = device_config.get("rate_changes", [0.0, 1.0])
                value_range = device_config.get("value_range", [0.0, 1.0])

                for row in df.iloc[:row_count].itertuples(index=False):
                    for index, value in enumerate(row):
                        for device_id in range(1, device_replicas + 1):
                            device_node = f"/Objects/{namespace_index}:{device}_{device_id}"
                            var_node = f"{device_node}/{all_variables[index]}"
                            var = await client.nodes.root.get_child(var_node)
                            if device_replicas > 1:
                                alt_value = max(
                                    value_range[0],
                                    min(
                                        value_range[1],
                                        value
                                        + random.uniform(
                                            rate_changes[0], rate_changes[1]
                                        ),
                                    ),
                                )
                                await var.set_value(float(alt_value))
                            else:
                                await var.set_value(float(value))
                    time.sleep(interval)

    async def read_csv_file(self, client: Client, namespace: str) -> None:
        """
        Read data from a CSV file and publish it to the OPC UA server.
        :param client: The AsyncIO OPC UA client.
        :param namespace: The namespace to which the device belongs.
        """
        namespace_index = await client.get_namespace_index(namespace)
        namespace_config = self.simulator_namespaces.get(namespace, {})
        interval = namespace_config.get("interval", 2)
        devices = namespace_config.get("devices", {})

        tasks = []
        for _, (device, device_config) in enumerate(devices.items(), start=1):
            task = asyncio.create_task(
                self.publish_data_for_device(
                    client, namespace_index, device, device_config, interval
                )
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def connect_with_retry(self, client: Client) -> None:
        """
        Connect to the OPC UA server with retry logic.
        :param client: The AsyncIO OPC UA client.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                await client.connect()
                break
            except Exception:
                retries += 1
                print(
                    f"Connection failed. Retrying in {self.retry_interval} seconds..."
                )
                time.sleep(self.retry_interval)

        if retries == self.max_retries:
            raise RuntimeError(
                f"Failed to connect to OPC UA server after {self.max_retries} retries."
            )

    async def run_simulation(self) -> None:
        """
        Run the OPC UA simulation.
        """
        client = Client(url=self.opcua_endpoint_url)
        await self.connect_with_retry(client)

        while True:
            for namespace, _ in self.simulator_namespaces.items():
                await self.read_csv_file(client, namespace)

        await client.disconnect()


if __name__ == "__main__":
    config_file_path = "/config/simulator_config.yml"
    simulator = OPCUASimulator(config_file_path)
    simulator.load_configuration()
    asyncio.run(simulator.run_simulation())
