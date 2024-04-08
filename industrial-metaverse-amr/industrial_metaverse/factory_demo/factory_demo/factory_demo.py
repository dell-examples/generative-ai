# Created by Scalers AI for Dell Inc.

import json
import os
import threading
import time

import carb
import cv2
import ffmpeg
import numpy as np
import yaml
import zenoh
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import (
    create_prim,
    define_prim,
    get_prim_at_path,
)
from omni.isaac.sensor.scripts.camera import Camera
from pxr import Gf


class FactoryDemo:
    def __init__(self):
        """
        Initialize class variables.
        """
        robot_values, zone_values = self.read_config()
        self.no_of_robots = 2

        self.user = os.getenv("USER")
        self.isaac_sim_pkg = os.getenv("ISAAC_SIM_PACKAGE")

        # Extract information about each robot.
        self.robot_topic = []
        self.stream_url = []
        self.robot_name = []
        self.robot_ids = []
        self.camera_name = []
        self.zones = []

        for key in robot_values:
            self.robot_ids.append(key)
            self.robot_topic.append(f"{key}/{robot_values[key]['robot_name']}")
            self.stream_url.append(robot_values[key]["url"])
            self.robot_name.append(robot_values[key]["robot_name"])
            self.camera_name.append(robot_values[key]["camera_name"])
            self.broker = robot_values[key]["broker"]
        for key in zone_values:
            self.zones.append(zone_values[key])

        self.counter1 = 0
        self.counter2 = 0

        # Configure waypoints for each robot to move in the world.
        self.waypoint1 = [(0, -19), (0, 0)]
        self.waypoint2 = [(0, -21), (0, 0)]

        self.CARTER_URL = (
            "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0/Isaac/Samples/Replicator/OmniGraph/carter_v2_nav_only.usd"
        )
        self.stop_thread = False
        self.frame_threads = [
            threading.Thread(target=self.frame_processing_thread, args=(i,))
            for i in range(1, self.no_of_robots + 1)
        ]

        # Define ffmpeg process to push frames from camera streams in isaac sim.
        self.ffmpeg_processes = [
            ffmpeg.input(
                "pipe:", format="rawvideo", pix_fmt="bgr24", s="1920x1080"
            )
            .output(
                self.stream_url[i - 1],
                f="rtsp",
                vcodec="libx264",
                pix_fmt="yuv420p",
                rtsp_transport="tcp",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
            for i in range(1, self.no_of_robots + 1)
        ]

        # Define publishers to publish robot information on zenoh.
        self.pubs = []
        for i in self.robot_topic:
            broker = f"tcp/{self.broker}:7445"
            self.pubs.append(
                self.establish_zenoh_connection(broker, i, max_retries=3)
            )

    def read_config(self):
        """
        Method to read config file.
        """
        dev_data = os.path.join(
            os.getenv("CONFIGDIR", ""), "metaverse_config.yaml"
        )
        with open(dev_data, "r") as c_file:
            try:
                config = yaml.safe_load(c_file)
            except yaml.YAMLError as err:
                err_msg = f"Error while parsing config file: {err}."
                self.logger.error(err_msg)
        return config["robot_config"], config["zone_config"]

    def establish_zenoh_connection(self, broker, key, max_retries=3):
        """
        Establishes a connection with Zenoh broker.
        """
        retries = 0
        session = None
        pub = None

        while retries < max_retries:
            try:
                zenoh_config = zenoh.Config()
                zenoh_config.insert_json5(
                    zenoh.config.CONNECT_KEY, json.dumps([broker])
                )
                zenoh_config.insert_json5(
                    "scouting/multicast/enabled", "false"
                )
                session = zenoh.open(zenoh_config)
                pub = session.declare_publisher(key)
                print("Zenoh broker connection established successfully.")
                return pub
            except Exception as e:
                print(e)
                if retries < max_retries - 1:
                    print(
                        f"Retrying to get the Zenoh broker connection ({retries + 1}/{max_retries})"
                    )
                retries += 1
                if retries < max_retries:
                    time.sleep(5)

        print(
            f"Zenoh broker connection cannot be established after {max_retries} retries. Exiting."
        )
        if session is not None:
            session.close()
        exit(1)

    def frame_processing_thread(self, idx):
        """
        Method which spawns as a seperate thread based on the number of cameras.
        For each robot this method pushes frames to rtsp server.
        It also pushes robot location information over zenoh to use on dashboard.
        """
        camera = self.cameras[idx - 1]
        ffmpeg_process = self.ffmpeg_processes[idx - 1]
        while not self.stop_thread:
            try:
                frame = camera.get_current_frame()
                if "rgba" in frame:
                    try:
                        image_data = frame[
                            "rgba"
                        ]  # extract rgba data from camera.
                        if idx < 2:
                            # idx 1 refers to robot 1 and extracts the location information of it.
                            carter_loc = self._carter_chassis1.GetAttribute(
                                "xformOp:translate"
                            ).Get()
                            if carter_loc[1] > -6.5:
                                map_pt = abs(carter_loc[1]) * 33.6 + 345
                                message = f"{self.robot_ids[idx-1]} in {self.zones[idx-1]},{int(map_pt)}"
                                loc = self.zones[idx - 1]
                            else:
                                map_pt = abs(carter_loc[1]) * 33.6 + 345
                                message = f"{self.robot_ids[idx-1]} in {self.zones[idx]},{int(map_pt)}"
                                loc = self.zones[idx]
                            text = f"Camera ID : {self.camera_name[idx-1]}\nNova ID : {self.robot_ids[idx-1]}\nZone ID : {loc}"
                        else:
                            # idx 2 refers to robot 2 and extracts the location information of it.
                            carter_loc = self._carter_chassis2.GetAttribute(
                                "xformOp:translate"
                            ).Get()
                            if carter_loc[1] > -8.5:
                                map_pt = abs(carter_loc[1]) * 37 + 224
                                message = f"{self.robot_ids[idx-1]} in {self.zones[idx]},{int(map_pt)}"
                                loc = self.zones[idx]
                            else:
                                map_pt = abs(carter_loc[1]) * 37 + 224
                                message = f"{self.robot_ids[idx-1]} in {self.zones[idx+1]},{int(map_pt)}"
                                loc = self.zones[idx + 1]
                            text = f"Camera ID :{self.camera_name[idx-1]}\nNova ID : {self.robot_ids[idx-1]}\nZone ID : {loc}"

                        # post process on the camera frame with location information.
                        image = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR)
                        cv2.rectangle(
                            image, (0, 0), (630, 150), (255, 255, 255), -1
                        )
                        y0, dy = 30, 35
                        for i, line in enumerate(text.split("\n")):
                            y = y0 + i * dy
                            cv2.putText(
                                image,
                                line,
                                (30, y),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (0, 0, 0),
                                2,
                            )

                        # writes framae to rtsp and publishes location data for the robot.
                        ffmpeg_process.stdin.write(image)
                        self.pubs[idx - 1].put(message)

                    except cv2.error:
                        pass
            except:
                pass

    def setup_scene(self):
        """
        Method which loads the usd files that are needed to setup the simulation.
        """
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # Load warehouse prim from nvidia.
        prim = get_prim_at_path("/World/Warehouse")
        if not prim.IsValid():
            prim = define_prim("/World/Warehouse", "Xform")
            asset_path = (
                assets_root_path
                + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
            )
            prim.GetReferences().AddReference(asset_path)

        # Load a custom warehouse if .usd file is available.
        # self.warehouse = create_prim(
        #     prim_path="/World/Warehouse",
        #     prim_type="Xform",
        #     position=np.array([0.0, 0.0, 0.0]),
        #     usd_path="/home/user/Desktop/Warehouse.usd"
        # )

        # Loads the custom factory floor.

        self.factory = create_prim(
            prim_path="/World/factory",
            prim_type="Xform",
            position=np.array([0.0, 0.0, 0.0]),
            usd_path=f"/home/{self.user}/.local/share/ov/pkg/{self.isaac_sim_pkg}/extension_examples/user_examples/factory_demo.usd",
        )

        # Load the robots.
        self.carter = create_prim(
            prim_path=f"/World/{self.robot_name[0]}",
            prim_type="Xform",
            position=np.array([-6.0, 12.0, 0.0]),
            usd_path=self.CARTER_URL,
        )

        self.carter = create_prim(
            prim_path=f"/World/{self.robot_name[1]}",
            prim_type="Xform",
            position=np.array([6, 14.0, 0.0]),
            usd_path=self.CARTER_URL,
        )

        # Get the target and chassis_link attributes from the 1st robots prim for further operations.
        self._carter_nav_target1 = self._world.stage.GetPrimAtPath(
            f"/World/{self.robot_name[0]}/targetXform"
        )
        self._carter_chassis1 = self._world.stage.GetPrimAtPath(
            f"/World/{self.robot_name[0]}/chassis_link"
        )
        self._carter_chassis1.GetAttribute("xformOp:translate").Set(
            (-6, 12, 0)
        )
        self.set_pose1()

        # Get the target and chassis_link attributes from the 2nd robots prim for further operations.
        self._carter_nav_target2 = self._world.stage.GetPrimAtPath(
            f"/World/{self.robot_name[1]}/targetXform"
        )
        self._carter_chassis2 = self._world.stage.GetPrimAtPath(
            f"/World/{self.robot_name[1]}/chassis_link"
        )
        self._carter_chassis2.GetAttribute("xformOp:translate").Set((6, 14, 0))
        self.set_pose2()

        # Define the cameras needed with their placement and orientation.
        cameras = [
            (
                f"/World/{self.robot_name[0]}/chassis_link/{self.camera_name[0]}",
                [0, 0, 1],
                None,
            ),
            (
                f"/World/{self.robot_name[1]}/chassis_link/{self.camera_name[1]}",
                [0, 0, 1],
                None,
            ),
        ]

        # Add the cameras to the world.
        self.cameras = [
            self._world.scene.add(
                Camera(
                    prim_path=path,
                    name=self.camera_name[i],
                    frequency=20,
                    resolution=(1920, 1080),
                    translation=translation,
                    orientation=orientation,
                )
            )
            for i, (path, translation, orientation) in enumerate(cameras)
        ]

        for camera in self.cameras:
            camera.set_visibility(False)
            camera.set_projection_type("fisheyePolynomial")

        for thread in self.frame_threads:
            thread.daemon = True
            thread.start()

    def set_pose1(self):
        """
        For the 1 step before the simulation starts this method is called to set the target value of 1st robot.
        """
        x, y = self.waypoint1[self.counter1]
        self._carter_nav_target1.GetAttribute("xformOp:translate").Set(
            (x, y, 0)
        )

    def set_pose2(self):
        """
        For the 1 step before the simulation starts this method is called to set the target value of 2nd robot.
        """
        x, y = self.waypoint2[self.counter2]
        self._carter_nav_target2.GetAttribute("xformOp:translate").Set(
            (x, y, 0)
        )

    async def setup_post_load(self):
        """
        After loading all prims this method is called to start the simulation and the physics callback is defined here.
        """
        self._world.add_physics_callback(
            "amr", callback_fn=self.on_physics_step
        )
        await self._world.play_async()

    def on_physics_step(self, step_size):
        """
        Physics callback method for when the simulation moves 1 step ahead.
        """

        # Checks whether amr 1 has reached its location and initiates the movement toward next location if destination is reached.
        carter_loc = self._carter_chassis1.GetAttribute(
            "xformOp:translate"
        ).Get()
        dest = self.waypoint1[self.counter1]

        if (
            Gf.Vec2f(dest[0], dest[1]) - Gf.Vec2f(carter_loc[0], carter_loc[1])
        ).GetLength() < 1:
            if len(self.waypoint1) - 1 == self.counter1:
                self.counter1 = -1
            self.counter1 += 1
            x, y = self.waypoint1[self.counter1]
            self._carter_nav_target1.GetAttribute("xformOp:translate").Set(
                (x, y, 0)
            )

        # Checks whether amr 2 has reached its location and initiates the movement toward next location if destination is reached.
        carter_loc2 = self._carter_chassis2.GetAttribute(
            "xformOp:translate"
        ).Get()
        dest2 = self.waypoint2[self.counter2]

        if (
            Gf.Vec2f(dest2[0], dest2[1])
            - Gf.Vec2f(carter_loc2[0], carter_loc2[1])
        ).GetLength() < 1:
            if len(self.waypoint2) - 1 == self.counter2:
                self.counter2 = -1
            self.counter2 += 1
            x, y = self.waypoint2[self.counter2]
            self._carter_nav_target2.GetAttribute("xformOp:translate").Set(
                (x, y, 0)
            )

    async def load_world_async(self):
        """
        When load buttom clicked on isaac sim this method is called to setup the simulation and run it.
        """
        self._world = World(stage_units_in_meters=1.0)
        await self._world.initialize_simulation_context_async()
        self.setup_scene()
        await self._world.pause_async()
        await self.setup_post_load()
        self._world.add_physics_callback("tasks_step", self._world.step_async)
        self._world.reset()
