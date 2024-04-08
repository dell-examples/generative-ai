# Created by Scalers AI for Dell Inc.

import asyncio
import os
import pickle
import time

import numpy as np
import yaml
from asyncua import Client, Node


class CompressorInferece:
    def __init__(self, config_file_path: str) -> None:
        """
        Initialize the OPCUASimulator with the configuration file path.
        :param config_file_path: Path to the YAML configuration file.
        """
        self.config_file_path = config_file_path
        self.opcua_endpoint_url = None
        self.simulator_namespaces = {}
        self.retry_interval = 5
        self.lr_model_path = "/model/bearing_fault.pkl"
        self.model = self.load_model()
        self.classes = ["Normal", "Cage Fault", "Ball Fault", "Outer Race"]

    def load_model(self):
        """Load the bearing fault LR Model."""
        with open(self.lr_model_path, "rb") as model_file:
            model = pickle.load(model_file)

        return model

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

    async def run_inference(self, client: Client, namespace: str) -> None:
        """
        Read data from a CSV file and publish it to the OPC UA server.
        :param client: The AsyncIO OPC UA client.
        :param namespace: The namespace to which the device belongs.
        """
        namespace_index = await client.get_namespace_index(namespace)
        namespace_config = self.simulator_namespaces.get(namespace, {})
        interval = namespace_config.get("interval", 1)

        devices = namespace_config.get("devices", {})
        failed_devices = 0

        for _, (device, device_config) in enumerate(devices.items(), start=1):
            sensor_nodes = device_config.get("sensor_nodes", [])
            device_replicas = device_config.get("replicas", 1)

            # read sensor nodes details of the device
            infer_input = []
            for sensor_name, sensor_config in sensor_nodes.items():
                for variable in sensor_config["variables"]:
                    var_node = f"{int(namespace_index)}:{sensor_name}/{int(namespace_index)}:{variable}"
                    for device_id in range(1, device_replicas + 1):
                        device_node = (
                            f"/Objects/{namespace_index}:{device}_{device_id}"
                        )
                        var_node = f"{device_node}/{var_node}"
                        var = await client.nodes.root.get_child(var_node)
                        infer_input.append(await var.read_value())

            # run inference
            input_data = np.array(infer_input)
            prediction = self.model.predict(input_data.reshape(1, -1))[0]
            # class_name = self.classes[prediction]
            failed_devices += 1 if prediction >= 1 else 0

        # publish compressor level details
        line = (
            "compressor,device=all "
            f"failed_devices={failed_devices},"
            f"is_failed={failed_devices>=1} "
            f"{int(time.time() * 1000000000)} \n"
        )
        print(line)
        time.sleep(interval)

    async def connect_with_retry(self, client: Client) -> None:
        """
        Connect to the OPC UA server with retry logic.
        :param client: The AsyncIO OPC UA client.
        """
        await client.connect()

    async def run_simulation(self) -> None:
        """
        Run the OPC UA simulation.
        """
        client = Client(url=self.opcua_endpoint_url)
        await self.connect_with_retry(client)

        while True:
            for namespace, _ in self.simulator_namespaces.items():
                await self.run_inference(client, namespace)

        await client.disconnect()


if __name__ == "__main__":
    config_file_path = "/config/simulator_config.yml"
    simulator = CompressorInferece(config_file_path)
    simulator.load_configuration()
    asyncio.run(simulator.run_simulation())
