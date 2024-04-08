# Created by Scalers AI for Dell Inc.

import copy
import json
import os
import random
import time
from datetime import datetime

import cv2
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yaml
import zenoh
from influxdb import InfluxDBClient
from zenoh import Reliability, Sample


class MetaverseDashboard:
    """
    Creates a Metaverse Dashboard class.
    """

    def __init__(self):
        """
        Initializes the MetaverseDashboard variables.
        """
        self.server_ip = os.getenv("SERVER_IP", "localhost")
        self.dashboard_ip = os.getenv("DASHBOARD_IP", "localhost")

        robot_values, zone_values = self.read_config()

        self.robot_topic = []
        self.robot_name = []
        self.robot_ids = []
        self.zones = []

        for key in robot_values:
            self.robot_ids.append(key)
            self.robot_topic.append(f"{key}/{robot_values[key]['robot_name']}")
            self.robot_name.append(robot_values[key]["robot_name"])
            self.broker = robot_values[key]["broker"]
        for key in robot_values:
            self.robot_topic.append(robot_values[key]["spill_topic"])
        for key in zone_values:
            self.zones.append(zone_values[key])

        self.previous_logs = []
        self.previous_logs2 = []
        self.message1 = ""
        self.message2 = ""
        self.message3 = False
        self.message4 = False

        self.color = (0, 255, 0, 128)
        self.x_nova1 = 140
        self.x_nova2 = 550
        self.total_a = 1164
        self.ref_point_a = 365
        self.total_b = 1450
        self.ref_point_b = 240

        self.incident_count = 0
        self.new_incident1 = True
        self.new_incident2 = True
        self.spills = []
        self.dev_fail_status = False

        self.prev_zone1 = self.zones[0]
        self.prev_zone2 = self.zones[2]
        self.ticket_id1 = "".join(
            random.choice("0123456789ABCDEF") for i in range(6)
        )
        self.prev_ticket_id1 = self.ticket_id1
        self.ticket_id2 = "".join(
            random.choice("0123456789ABCDEF") for i in range(6)
        )
        self.prev_ticket_id2 = self.ticket_id2

        self.influx_client = InfluxDBClient(
            host=f"{self.server_ip}", port=8086, database="factory"
        )

        self.image = cv2.imread("/src/assets/metaverse_map.png")
        b_channel, g_channel, r_channel = cv2.split(self.image)
        self.image = cv2.merge((r_channel, g_channel, b_channel))
        self.alert = cv2.imread("/src/assets/alert.png")
        self.alert = cv2.resize(self.alert, (50, 50))
        b_channel, g_channel, r_channel = cv2.split(self.alert)
        self.alert = cv2.merge((r_channel, g_channel, b_channel))

        broker = f"tcp/{self.broker}:7445"
        self.session = self.establish_zenoh_connection(broker, max_retries=3)
        self.subs = []
        self.create_zen_con()

    def create_zen_con(self):
        """
        creates subscribers for each key.
        """
        for key in self.robot_topic:
            self.subs.append(
                self.session.declare_subscriber(
                    key, self.listener, reliability=Reliability.RELIABLE()
                )
            )

    def establish_zenoh_connection(self, broker, max_retries=3):
        """
        Establishes a connection with the Zenoh broker.
        """
        retries = 0
        session = None

        while retries < max_retries:
            try:
                zenoh_config = zenoh.Config()
                zenoh_config.insert_json5(
                    zenoh.config.LISTEN_KEY, json.dumps([broker])
                )
                zenoh_config.insert_json5(
                    "scouting/multicast/enabled", "false"
                )
                session = zenoh.open(zenoh_config)

                return session
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

    def listener(self, sample: Sample):
        """
        Listener method for receiving robot information.
        """
        key = sample.key_expr
        message = sample.payload.decode("utf-8")
        if key == "NOVA_CARTER_B1/CarterNav1":
            self.message1 = message.split(",")[0]
            self.y_nova1 = int(message.split(",")[1])
        if key == "NOVA_CARTER_B2/CarterNav2":
            self.message2 = message.split(",")[0]
            self.y_nova2 = int(message.split(",")[1])
        if key == "cam1":
            self.message3 = True
        if key == "cam2":
            self.message4 = True

    def read_config(self):
        """
        Method to read config file.
        """
        dev_data = "/src/config/metaverse_config.yaml"
        with open(dev_data, "r") as c_file:
            try:
                config = yaml.safe_load(c_file)
            except yaml.YAMLError as err:
                err_msg = f"Error while parsing config file: {err}."
                self.logger.error(err_msg)
        return config["robot_config"], config["zone_config"]

    def sensor_data(self):
        """
        Method to read the sensor data from influxdb and send the graph information for dashboard.
        """
        q1 = "SELECT time,failed_devices,is_failed FROM compressor WHERE device = 'all' AND time > now() - 3m ORDER BY time"
        result = self.influx_client.query(q1)
        temp = result.raw["series"][0]
        self.df = pd.DataFrame(temp["values"], columns=temp["columns"])
        self.dev_fail_status = bool(list(self.df["is_failed"])[-1])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.df["time"],
                y=self.df["failed_devices"],
                mode="lines",
                line=dict(shape="spline", smoothing=1),
                name="Random Values",
            )
        )
        fig.update_layout(
            paper_bgcolor="black",
            plot_bgcolor="black",
            font_color="#c934eb",
            height=160,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        return fig

    def graph(self):
        """
        Create graph for incidents over time to display on dashboard
        """
        fig2 = go.Figure()
        self.spills.append(self.incident_count)
        self.spills = self.spills[-180:]
        fig2.add_trace(
            go.Scatter(
                x=self.df["time"],
                y=self.spills,
                mode="lines",
                line=dict(shape="spline", smoothing=1),
                name="Incident Count",
            )
        )
        fig2.update_layout(
            paper_bgcolor="black",
            plot_bgcolor="black",
            font_color="#c934eb",
            height=160,
        )
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=False)
        return fig2

    def append_and_format_log(self):
        """
        Format logs for robot 1 and trigger incident notifications.
        """
        if self.message1.split("in ")[1] != self.prev_zone1:
            self.ticket_id1 = "".join(
                random.choice("0123456789ABCDEF") for i in range(6)
            )
            self.prev_zone1 = self.message1.split("in ")[1]
        if self.message3:
            self.previous_logs.append(
                self.message1
                + f"\ndetected a chemical spill. Incident ID : {self.ticket_id1}"
            )
            if self.ticket_id1 != self.prev_ticket_id1:
                self.incident_count += 1
                self.prev_ticket_id1 = self.ticket_id1
            self.color1 = (255, 0, 0, 128)
        else:
            self.previous_logs.append(self.message1)
            self.color1 = (0, 255, 0, 128)
        if len(self.previous_logs) > 5:
            self.previous_logs = self.previous_logs[-5:]
        markdown_logs = "<div style='background-color: black; padding: 10px;'><pre style='color: white;'>"
        if self.message3:
            markdown_logs += "\n".join(
                [
                    f"{datetime.now().strftime('%m/%d')} {datetime.now().hour}:{datetime.now().minute} - {log}"
                    for log in self.previous_logs[:-1]
                ]
            )
            markdown_logs += f"\n{datetime.now().strftime('%m/%d')} {datetime.now().hour}:{datetime.now().minute} - <span style='color: red;'>{self.previous_logs[-1]}</span>"
        else:
            markdown_logs += "\n".join(
                [
                    f"{datetime.now().strftime('%m/%d')} {datetime.now().hour}:{datetime.now().minute} - {log}"
                    for log in self.previous_logs
                ]
            )
        markdown_logs += "</pre></div>"
        self.message3 = False

        return markdown_logs

    def append_and_format_log2(self):
        """
        Format logs for robot 2 and trigger incident notifications.
        """
        if self.message2.split("in ")[1] != self.prev_zone2:
            self.ticket_id2 = "".join(
                random.choice("0123456789ABCDEF") for i in range(6)
            )
            self.prev_zone2 = self.message2.split("in ")[1]
        if self.message4:
            self.previous_logs2.append(
                self.message2
                + f"\ndetected a chemical spill. Incident ID : {self.ticket_id2}"
            )
            if self.ticket_id2 != self.prev_ticket_id2:
                self.incident_count += 1
                self.prev_ticket_id2 = self.ticket_id2
            self.color2 = (255, 0, 0, 128)
        else:
            self.previous_logs2.append(self.message2)
            self.color2 = (0, 255, 0, 128)
        if len(self.previous_logs2) > 5:
            self.previous_logs2 = self.previous_logs2[-5:]
        markdown_logs = "<div style='background-color: black; padding: 10px;'><pre style='color: white;'>"
        if self.message4:
            markdown_logs += "\n".join(
                [
                    f"{datetime.now().strftime('%m/%d')} {datetime.now().hour}:{datetime.now().minute} - {log}"
                    for log in self.previous_logs2[:-1]
                ]
            )
            markdown_logs += f"\n{datetime.now().strftime('%m/%d')} {datetime.now().hour}:{datetime.now().minute} - <span style='color: red;'>{self.previous_logs2[-1]}</span>"
        else:
            markdown_logs += "\n".join(
                [
                    f"{datetime.now().strftime('%m/%d')} {datetime.now().hour}:{datetime.now().minute} - {log}"
                    for log in self.previous_logs2
                ]
            )
        markdown_logs += "</pre></div>"
        self.message4 = False

        return markdown_logs

    def calculate_completion_percentage_a(self, distance_moved):
        """
        Calculate % completion of the 1st robots inspection cycle.
        """
        distance_traveled = abs(distance_moved)
        completion_percentage = (distance_traveled / self.total_a) * 100
        if completion_percentage == 100:
            completion_percentage = 0
            return 100
        return completion_percentage

    def calculate_completion_percentage_b(self, distance_moved):
        """
        Calculate % completion of the 2nd robots inspection cycle.
        """
        distance_traveled = abs(distance_moved)
        completion_percentage = (distance_traveled / self.total_b) * 100
        if completion_percentage == 100:
            completion_percentage = 0
            return 100
        return completion_percentage

    def image_generator(self):
        """
        Method to generate the map with live data from all sensors.
        """
        image_copy = copy.deepcopy(self.image)
        overlay = image_copy.copy()
        self.color = (0, 255, 0, 128)
        thickness = cv2.FILLED
        if self.message1 == f"{self.robot_ids[0]} in {self.zones[0]}":
            cv2.rectangle(
                image_copy, (364, 0), (0, 514), self.color1, thickness
            )
        if self.message1 == f"{self.robot_ids[0]} in {self.zones[1]}":
            cv2.rectangle(
                image_copy, (364, 558), (0, 1106), self.color1, thickness
            )
        if self.message2 == f"{self.robot_ids[1]} in {self.zones[2]}":
            cv2.rectangle(
                image_copy, (696, 0), (409, 516), self.color2, thickness
            )
        if self.message2 == f"{self.robot_ids[1]} in {self.zones[3]}":
            cv2.rectangle(
                image_copy, (696, 564), (410, 1106), self.color2, thickness
            )
        final_image = cv2.addWeighted(overlay, 0.5, image_copy, 1 - 0.5, 0)
        distance_moved_a = self.y_nova1 - self.ref_point_a
        completion_percentage_a = self.calculate_completion_percentage_a(
            distance_moved_a
        )
        if self.ref_point_a == 947:
            completion_percentage_a = completion_percentage_a + 50
        if completion_percentage_a == 50:
            self.ref_point_a = 947
        if completion_percentage_a > 99:
            self.ref_point_a = 365
        cv2.circle(
            final_image, (self.x_nova1, self.y_nova1), 15, (255, 255, 0), -1
        )
        cv2.putText(
            final_image,
            f"({int(completion_percentage_a)}%)",
            (self.x_nova1 + 10, self.y_nova1),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            final_image,
            f"{self.robot_ids[0]}",
            (self.x_nova1, self.y_nova1 + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        distance_moved_b = self.y_nova2 - self.ref_point_b
        completion_percentage_b = self.calculate_completion_percentage_b(
            distance_moved_b
        )
        if self.ref_point_b == 974:
            completion_percentage_b = completion_percentage_b + 50
        if completion_percentage_b == 50:
            self.ref_point_b = 974
        if completion_percentage_b > 99:
            self.ref_point_b = 240
        cv2.putText(
            final_image,
            f"({int(completion_percentage_b)}%)",
            (self.x_nova2 + 10, self.y_nova2),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.circle(
            final_image, (self.x_nova2, self.y_nova2), 15, (255, 255, 0), -1
        )
        cv2.putText(
            final_image,
            f"{self.robot_ids[1]}",
            (self.x_nova2, self.y_nova2 + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        if self.dev_fail_status:
            final_image[1053 : 1053 + 50, 633 : 633 + 50] = self.alert

        return final_image

    def create_dashboard(self):
        """
        Creates the Metaverse dashboard using Gradio.
        """
        with gr.Blocks(title="Manufacturing Factory Console") as demo:
            with gr.Row(elem_id="row"):
                with gr.Column(scale=2, elem_id=""):
                    with gr.Box(elem_id="stream"):
                        iframe = gr.HTML(
                            value=f"<iframe src='http://{self.dashboard_ip}:9000/' width='970px' height='275px'></iframe>",
                            height=560,
                        )
                    with gr.Row(elem_id="row3"):
                        with gr.Box(elem_id="box3"):
                            with gr.Column():
                                with gr.Row():
                                    sensor = gr.Plot(
                                        label="Compressor failures over time",
                                        show_label=True,
                                        elem_id="stock",
                                    )
                                    incident_count = gr.Plot(
                                        label="Incident count over time",
                                        show_label=True,
                                        elem_id="stock",
                                    )
                                with gr.Row():
                                    sweat_alert = gr.Markdown(value="")
                                    sweat_alert2 = gr.Markdown(value="")
                with gr.Column(elem_id="col2"):
                    with gr.Row(elem_id="row1"):
                        with gr.Box(elem_id="box1"):
                            with gr.Column():
                                with gr.Row():
                                    map = gr.Image(
                                        value="/src/assets/metaverse_map.png"
                                    )

            demo.load(
                self.append_and_format_log,
                inputs=[],
                outputs=[sweat_alert],
                every=5,
            )
            demo.load(
                self.append_and_format_log2,
                inputs=[],
                outputs=[sweat_alert2],
                every=5,
            )
            demo.load(
                self.image_generator,
                inputs=[],
                outputs=[map],
                every=1,
            )
            demo.load(
                self.sensor_data,
                inputs=[],
                outputs=[sensor],
                every=1,
            )
            demo.load(
                self.graph,
                inputs=[],
                outputs=[incident_count],
                every=5,
            )
        return demo

    def run(self):
        """
        Runs the Metaverse Dashboard.
        """
        demo = self.create_dashboard()
        demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    dashboard = MetaverseDashboard()
    dashboard.run()
