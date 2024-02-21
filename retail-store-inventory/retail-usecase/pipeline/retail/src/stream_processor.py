# Created by Scalers AI for Dell Inc.

import json
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import zenoh
from draw_utils import DrawUtils


class StreamProcessor:
    def __init__(self, rtsp_url, key, broker, json_file, stream_name, logger):
        """
        Initializes the Inference object with necessary parameters.

        Parameters:
        rtsp_url (str): The RTSP URL for video streaming.
        key (str): Key for the Zenoh publisher.
        broker (str): Zenoh broker address.
        json_file (str): File containing JSON data for zones and items.
        stream_name (str): Name of the video stream.
        """
        self.stream_name = stream_name
        self.rtsp_url = rtsp_url
        self.key = key
        self.broker = broker
        self.json_file = json_file
        self.logger = logger
        self.cap = None
        self.open_video()

        try:
            self.model = hub.KerasLayer(
                "saved_model",
                signature="serving_default",
                signature_outputs_as_dict=True,
            )
        except:
            self.logger.error("Model File is not Available. Exiting.")
            sys.exit(1)

        self.zones = []
        self.objects = []
        if self.validate_json_file(self.json_file):
            self.logger.info(
                f"The JSON structure in {self.json_file} is valid."
            )
        else:
            self.logger.error(
                f"The JSON structure in {self.json_file} is not valid."
            )
            sys.exit(1)

        with open(self.json_file, "r") as f:
            self.k = json.load(f)
            self.zones = self.k["contours"]
            self.objects = self.k["items"]

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fps = self.cap.get(5)
        (
            self.session,
            self.pub,
            self.pub_mets,
        ) = self.establish_zenoh_connection(max_retries=3)

        self.frame_count = 0
        self.prev_fps = 0.0
        self.start_time = time.time()

        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "green": (0, 255, 0),
        }
        self.draw_utils = DrawUtils(
            self.zones,
            self.objects,
            self.stream_name,
            self.colors,
            self.logger,
        )

    def open_video(self):
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                self.logger.error("Unable to open the video source")
                sys.exit(1)
        except cv2.error as e:
            self.logger.error(f"OpenCV error: {e}")
        except Exception as e:
            self.logger.error(f"An error occurred:{e}")

    def release_video(self):
        if self.cap is not None:
            self.cap.release()

    def validate_json_file(self, json_file):
        """
        Validates the structure and contents of a JSON file.

        Parameters:
        json_file (str): File path of the JSON data.

        Returns:
        bool: True if the JSON structure is valid, otherwise False.
        """
        if not os.path.isfile(json_file):
            self.logger.error(f"The file {json_file} does not exist.")
            return False

        try:
            with open(json_file, "r") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                self.logger.error("JSON data is not a dictionary.")
                return False

            contours = data.get("contours", {})
            items = data.get("items", {})

            if not isinstance(contours, dict) or not isinstance(items, dict):
                self.logger.error(
                    "Invalid 'contours' or 'items' format. Both should be dictionaries."
                )
                return False

            contour_zones = set(contours.keys())
            item_zones = set(items.keys())

            if contour_zones != item_zones:
                self.logger.error(
                    "The zones in 'contours' and 'items' do not match."
                )
                return False

            for zone, points in contours.items():
                if not isinstance(points, list) or len(points) != 6:
                    self.logger.error(
                        f"Invalid number of points in '{zone}' (should have exactly 6 points)."
                    )
                    return False

                for point in points:
                    if (
                        not isinstance(point, dict)
                        or "x" not in point
                        or "y" not in point
                    ):
                        self.logger.error(f"Invalid point format in '{zone}'.")
                        return False

                    if not isinstance(point["x"], int) or not isinstance(
                        point["y"], int
                    ):
                        self.logger.error(
                            f"Invalid data types for 'x' and 'y' values in '{zone}'. Both should be integers."
                        )
                        return False

            for zone, value in items.items():
                if not isinstance(value, int):
                    self.logger.error(
                        f"Invalid value format for '{zone}' in 'items'. It should be an integer."
                    )
                    return False

            return True

        except json.JSONDecodeError:
            self.logger.error("JSON decoding failed.")
            return False

    def establish_zenoh_connection(self, max_retries=3):
        """
        Establishes a connection with Zenoh broker.

        Parameters:
        max_retries (int): Maximum number of connection retries (default: 3).

        Returns:
        tuple: Session, publisher, and metadata publisher objects upon successful connection.
        """
        retries = 0
        session = None
        pub = None
        pub_mets = None

        while retries < max_retries:
            try:
                zenoh_config = zenoh.Config()
                zenoh_config.insert_json5(
                    zenoh.config.CONNECT_KEY, json.dumps([self.broker])
                )
                zenoh_config.insert_json5(
                    "scouting/multicast/enabled", "false"
                )
                session = zenoh.open(zenoh_config)
                pub = session.declare_publisher(self.key)
                pub_mets = session.declare_publisher("metadata")
                self.logger.info(
                    "Zenoh broker connection established successfully."
                )
                return session, pub, pub_mets
            except Exception:
                if retries < max_retries - 1:
                    self.logger.warning(
                        f"Retrying to get the Zenoh broker connection ({retries + 1}/{max_retries})"
                    )
                retries += 1
                if retries < max_retries:
                    time.sleep(5)

        self.logger.error(
            f"Zenoh broker connection cannot be established after {max_retries} retries. Exiting."
        )
        if session is not None:
            session.close()
        exit(1)

    def preprocess_image(self, target_size=(300, 300)):
        """
        Preprocesses an image by resizing it to the specified target size.

        Parameters:
            - self: The current instance of the class containing this method.
            - target_size (tuple): A tuple specifying the desired width and height of the resized image.

        Returns:
            - image: The preprocessed image, resized to the target size.

        """
        image = cv2.resize(self.frame, target_size)
        return image

    def process_rtsp_video_stream(self):
        """
        Captures and processes frames from an RTSP video stream.

        This function continuously captures frames from an RTSP video stream, performs inference on each frame,
        draws bounding boxes, calculates and displays frames per second (FPS), and publishes the processed frames
        and metadata to the appropriate channels. The function runs in a loop until the video stream ends.

        Note:
        - The function assumes that the video capture object (self.cap) is already opened and configured.
        - The inference model is expected to be available via self.infer_frame().
        - This function also relies on helper methods and attributes provided by the class.

        Returns:
            None

        Raises:
            SystemExit: If an exception occurs during inference, the program exits.

        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                try:
                    boxes, scores = self.infer_frame()

                    self.bounding_boxes = []
                    # Draw bounding boxes on the frame with confidence scores
                    for box, score in zip(boxes, scores):
                        ymin, xmin, ymax, xmax = box
                        xmin = int(xmin * frame.shape[1])
                        xmax = int(xmax * frame.shape[1])
                        ymin = int(ymin * frame.shape[0])
                        ymax = int(ymax * frame.shape[0])
                        self.bounding_boxes.append([xmin, ymin, xmax, ymax])

                except:
                    self.logger.error(f"Inference cannot be Done. Exiting.")
                    sys.exit(1)

                frame, metadata = self.draw_utils.draw_shelf_bounding_boxes(
                    self.frame, self.bounding_boxes
                )

                self.pub_mets.put(json.dumps(metadata))

                # # Increment the frame count
                self.frame_count += 1

                # Calculate FPS every second
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= 5.0:
                    fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()


                    self.logger.info(
                        f"Process for {self.key}: FPS = {fps:.2f}"
                    )
                    frame = self.draw_utils.draw_hollow_rectangle_with_text(
                        frame,
                        f"FPS: {fps:.2f}",
                        f"STREAM ID: {self.stream_name}",
                        (400, 30),
                        top_text_scale=1,
                        bottom_text_scale=1,
                        text_padding=20,
                    )
                    self.prev_fps = fps
                else:
                    frame = self.draw_utils.draw_hollow_rectangle_with_text(
                        frame,
                        f"FPS: {self.prev_fps:.2f}",
                        f"STREAM ID: {self.stream_name}",
                        (400, 30),
                        top_text_scale=1,
                        bottom_text_scale=1,
                        text_padding=20,
                    )

                image_buffer = np.asarray(frame).tobytes()
                self.pub.put(image_buffer)
            else:
                # If the video ends, set the video capture object back to the beginning
                self.release_video()
                self.open_video()
                continue

    def infer_frame(self):
        """
        Perform object detection inference on a preprocessed image.

        This function takes a preprocessed image as input, runs it through an object detection model, and
        post-processes the results to obtain a list of bounding boxes and their corresponding confidence scores.

        Returns:
            Tuple (filtered_boxes, filtered_scores):
            - filtered_boxes (List of Lists): List of bounding boxes in the format [ymin, xmin, ymax, xmax].
            - filtered_scores (List of floats): List of confidence scores corresponding to the detected objects.

        Note:
        - The object detection model is expected to be available via the 'self.model' attribute.
        - A confidence threshold of 0.53 is applied to filter out low-confidence detections.

        """
        input_image = self.preprocess_image()

        input_image = tf.convert_to_tensor(input_image, dtype=tf.uint8)
        input_image = tf.expand_dims(
            input_image, axis=0
        )  # Add batch dimension

        detections = self.model(input_image)

        # Post-process the detections
        boxes = detections["detection_boxes"][0].numpy()
        scores = detections["detection_scores"][0].numpy()

        # Filter out low-confidence detections
        threshold = 0.53

        filtered_boxes = []
        filtered_scores = []

        for i in range(len(scores)):
            if scores[i] > threshold:
                filtered_boxes.append(boxes[i])
                filtered_scores.append(scores[i])

        return filtered_boxes, filtered_scores

