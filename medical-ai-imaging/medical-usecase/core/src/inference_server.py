# Copyright 2022 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Scalers AI for Dell Inc.

import os
import sys

import amdinfer
import amdinfer.pre_post as pre_post
import cv2
import numpy as np
from resnet import parse_args, print_label


class Inference:
    def __init__(self, args):
        """
        Initialize the Inference object with arguments.
        """
        self.args = args
        self.client = None
        self.endpoint = None

    def preprocess(self, paths: list, input_size: tuple) -> list:
        """
        Given a list of paths to images, preprocess the images and return them

        Args:
            paths (list[str]): Paths to images
            input_size (tuple): Input size of the images

        Returns:
            list[numpy.ndarray]: List of preprocessed images
        """
        image_path = paths[0]
        image = cv2.imread(
            image_path, cv2.IMREAD_GRAYSCALE
        )  # Load the image in grayscale
        image = cv2.resize(
            image, (224, 224)
        )  # Resize the image to match the model's input shape
        image = image.astype("float32") / 255.0  # Normalize the pixel values

        # Add a batch dimension to the image
        input_data = np.expand_dims(image, axis=-1)  # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        return input_data

    def postprocess(self, output: np.ndarray, k: int) -> np.ndarray:
        """
        Perform postprocessing on the model output.

        Args:
            output (numpy.ndarray): Model output
            k (int): Value for postprocessing

        Returns:
            np.ndarray : Result of postprocessing
        """
        return pre_post.resnet50PostprocessFp32(output, k)

    def construct_requests(self, images: list) -> list:
        """
        Construct inference requests for a list of images.

        Args:
            images (list[numpy.ndarray]): List of images

        Returns:
            list[amdinfer.ImageInferenceRequest]: List of inference requests
        """
        requests = []
        for image in images:
            requests.append(amdinfer.ImageInferenceRequest(image))
        return requests

    def load(self) -> None:
        """
        Load the model for inference.
        """
        if not amdinfer.serverHasExtension(self.client, "tfzendnn"):
            print(
                "TF+ZenDNN is not enabled. Please recompile with it enabled to run this example"
            )
            sys.exit(0)

        if not self.args.model:
            print(
                "A path to the model on the server must be specified if loading a new worker"
            )
            print(
                "If your model is already loaded, then pass the endpoint to it with --endpoint"
            )
            print(
                "If your model needs loading, then pass the path to the model on the server with --model"
            )
            raise ValueError("No model argument")

        parameters = amdinfer.ParameterMap()
        parameters.put("model", self.args.model)
        parameters.put("input_size", self.args.input_size)
        parameters.put("output_classes", self.args.output_classes)
        parameters.put("input_node", self.args.input_node)
        parameters.put("output_node", self.args.output_node)
        parameters.put("batch_size", self.args.batch_size)
        self.endpoint = self.client.workerLoad("tfzendnn", parameters)
        amdinfer.waitUntilModelReady(self.client, self.endpoint)

    def infer(self, images: list) -> str:
        """
        Perform inference on the provided images.

        Args:
            images (list[numpy.ndarray]): Images to perform inference on

        Returns:
            str: Inference result
        """
        requests = self.construct_requests(images)
        print("Making inferences...")
        for request in requests:
            response = self.client.modelInfer(self.endpoint, request)
            assert not response.isError()
            outputs = response.getOutputs()
            assert len(outputs) == 1
            top_indices = self.postprocess(outputs[0], self.args.top)
            k = print_label(top_indices, self.args.labels)
            print("hello", k.split()[1])

            return k.split()[1]

    def main(self):
        server_addr = f"{self.args.ip}:{self.args.grpc_port}"
        self.client = amdinfer.GrpcClient(server_addr)
        if self.args.wait:
            pass
        elif self.args.ip == "127.0.0.1" and not self.client.serverLive():
            print("No server detected. Starting locally...")
            server = amdinfer.Server()
            server.startGrpc(self.args.grpc_port)
        elif not self.client.serverLive():
            raise ConnectionError(
                f"Could not connect to server at {server_addr}"
            )
        print("Waiting until the server is ready...")
        amdinfer.waitUntilServerReady(self.client)

        if self.args.endpoint:
            self.endpoint = self.args.endpoint
            if not self.client.modelReady(self.endpoint):
                raise ValueError(
                    f"Model at {self.endpoint} does not exist or isn't ready. Verify the endpoint or omit the --endpoint flag to load a new worker"
                )
        else:
            print("Loading worker...")
            self.load()


class ArgGetter:
    def get_args(self):
        args = parse_args()
        if (not args.model) and (not args.endpoint):
            root = os.getenv("AMDINFER_ROOT")
            assert root is not None
            args.model = (
                root
                + "/external/artifacts/resnet50/resnet_v1_50_baseline_6.96B_922.pb"
            )

        if not args.input_node:
            args.input_node = "input"

        if not args.output_node:
            args.output_node = "resnet_v1_50/predictions/Reshape_1"

        return args


if __name__ == "__main__":
    argClass = ArgGetter()
    args = argClass.get_args()
    inference = Inference(args)
    inference.main()
