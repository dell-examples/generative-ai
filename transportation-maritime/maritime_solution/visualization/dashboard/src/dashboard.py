# Created by Scalers AI for Dell Inc.

import os
import yaml
import sys
import gradio as gr
import logging
from dashboard_utils import DashboardUtils

class TransportationDashboard:
    """
    A class to create a Transportation Dashboard using Gradio and InfluxDB.
    """

    def __init__(self):
        """
        Initializes the TransportationDashboard object and establishes connection to the InfluxDB server.

        Attributes:
            IPAddr (str): The IP address of the InfluxDB server. Defaults to 'localhost' if not provided through the environment variable 'SERVER_IP'.
            client: The InfluxDB client used for database operations.
            connected (bool): Indicates whether the class is successfully connected to the InfluxDB server.
            utils: An instance of DashboardUtils for utility functions related to the transportation dashboard.
        """
        self.IPAddr = os.getenv("VISUALIZATION_SERVER_IP", "localhost")
        self.connected = False

        # Initialize logger
        logging.basicConfig(
            filename="app.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()

        self.config_path = "/src/simulator_config.yaml"

        # check if yaml file exists and is in correct format
        if self.validate_streams_yaml(self.config_path, self.logger):
            self.logger.info(f"The format of {self.config_path} is valid.")
        else:
            self.logger.error(f"The format of {self.config_path} is not valid.")
            sys.exit(1)

        self.utils = DashboardUtils()
    

    def validate_streams_yaml(self, yaml_file, logger):
        """
        Validates the format of a YAML configuration file.

        Parameters:
        yaml_file (str): File path of the YAML configuration.

        Returns:
        bool: True if the YAML format is valid, otherwise False.
        """
        if not os.path.isfile(yaml_file):
            logger.error(f"The file {yaml_file} does not exist.")
            sys.exit(1)

        try:
            with open(yaml_file, "r") as stream:
                data = yaml.safe_load(stream)

            if not isinstance(data, dict) or "rtsp_streams" not in data:
                return False

            cargo = data["cargo_info"]
            if not isinstance(cargo, dict):
                return False
            
            tot = cargo["total_containers"]
            if not isinstance(tot, int):
                return False

            ref = cargo["reefer_containers"]
            if not isinstance(ref, int):
                return False

            voyage = data["voyage_details"]
            if not isinstance(voyage, dict):
                return False

            vnum = voyage["voyage_number"]
            if not isinstance(vnum, str):
                return False

            vname = voyage["vessel_name"]
            if not isinstance(vname, str):
                return False

            aport = voyage["arrival_port"]
            if not isinstance(aport, str):
                return False

            dport = voyage["departure_port"]
            if not isinstance(dport, str):
                return False

            return True
    
        except yaml.YAMLError:
            return False

    def create_dashboard(self):
        """
        Creates the Transportation dashboard using Gradio.
        """
        with gr.Blocks(
            title="Transportation", css="website.css"
        ) as demo:
            with gr.Tab("Dashboard"):
                with gr.Row():
                    with gr.Column(scale=1.5):
                        with gr.Row():
                            with gr.Box():
                                with gr.Column():
                                    with gr.Row(elem_id="trip_info_row"):
                                        ship_icon = gr.Image(
                                            value=f"http://{self.IPAddr}:5000/static/ship.jpg",
                                            label = "Ship Icon",
                                            show_label = False,
                                            elem_id="ship_icon"
                                        )
                                        trip_details = gr.HTML(value = self.utils.get_trip_info(), every = 5)
                                        trip_status = gr.HTML(value = self.utils.get_trip_status, every = 5)
                        with gr.Row(scale=4):
                            with gr.Box(elem_id="stream"):
                                iframe = gr.HTML(
                                    value=f"<iframe src='http://{self.IPAddr}:9000/' width=100% height=100%></iframe>"
                                )
                        with gr.Row(elem_id="row2"):
                            with gr.Box(elem_id="box3"):
                                stock = gr.Plot(
                                    label="People Violations Timeseries Graph",
                                    show_label=True,
                                    elem_id="stock",
                                )
                    with gr.Column(elem_id="col2"):
                        with gr.Row():
                            with gr.Box(elem_id="box1"):
                                with gr.Column():
                                    with gr.Row(elem_id="map_row"):
                                        iframe = gr.HTML(
                                            value=f"<iframe src='http://{self.IPAddr}:5000/' width=100% height=100% id='mapFrame'></iframe>",
                                        )
                                    with gr.Row():
                                        curr_items = gr.HTML(value = self.utils.get_refeer_container_count)
                                        total_items = gr.Plot(
                                            label="Reefers Sweating Alert Count",
                                            show_label=True,
                                            elem_id="total_items",
                                        )
                                        power_alert = gr.Plot(
                                            label="Reefers Power Usage",
                                            show_label=True,
                                            elem_id="power_alert",
                                        )
                                    with gr.Row(elem_id="sweat_warning_row"):
                                        sweat_alert = gr.Dataframe(
                                            scale=3,
                                            headers=["Container ID", "Dew Point °C"],
                                            label = "Container Sweating Alert",
                                            show_label=True,
                                            wrap=True,
                                            interactive=False,
                                            row_count=(3, "fixed"),
                                            col_count=(2, "fixed"),
                                            value= lambda: self.utils.get_sweat_cont_info(self.utils.connect_to_database()),
                                            every=1,
                                            elem_id="sweat_info",
                                        )
                                        sweat_warning_sign = gr.Image(
                                            value=f"http://{self.IPAddr}:5000/static/normal_1.png",
                                            label = "Container Sweating Alert",
                                            show_label = True,
                                            every = 5,
                                            elem_id="sweat_warning"
                                        )
                                    with gr.Row():
                                        sweat_alert_plot = gr.Plot(
                                            label="Container Sweating Alert",
                                            show_label=True,
                                            elem_id="sweat_alert_plot",
                                        )
            with gr.Tab("Voyage Report"):
                with gr.Row(elem_id="report_row"):

                    # Display the Markdown content in the application
                    report_md = gr.Markdown(value = self.utils.get_report, every=5)

            demo.load(lambda: self.utils.get_worker_data(self.utils.connect_to_database()), inputs=[], outputs=[stock], every=1)
            demo.load(lambda: self.utils.get_alert_data(self.utils.connect_to_database()), inputs=[], outputs=[sweat_alert_plot], every=1)
            demo.load(self.utils.get_image_path_sweat, inputs=[], outputs=[sweat_warning_sign], every=1)
            demo.load(
                lambda: self.utils.generate_indicator(
                    self.utils.get_dew_count(self.utils.connect_to_database())
                ),
                inputs=[],
                outputs=[total_items],
                every=1,
            )
            demo.load(
                lambda: self.utils.create_gauage(self.utils.connect_to_database()),
                inputs=[],
                outputs=[power_alert],
                every=1,
            )
            demo.load(
                self.utils.get_sweat_alert,
                inputs=[],
                outputs=[sweat_alert],
                every=1,
            )
        return demo

    def run(self):
        """
        Runs the Transportation Dashboard.
        """
        demo = self.create_dashboard()
        demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    dashboard = TransportationDashboard()
    dashboard.run()
