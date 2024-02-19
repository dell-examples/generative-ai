# Created by Scalers AI for Dell Inc.

import os
import logging
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from influxdb import InfluxDBClient
from tabulate import tabulate

class DashboardUtils:

    def __init__(self):
        self.InfluxIPAddr = os.getenv("AI_SERVICES_SERVER_IP", "localhost")
        self.VisualIPAddr = os.getenv("VISUALIZATION_SERVER_IP", "localhost")
        self.client = None
        self.connected = False

        # Initialize logger
        logging.basicConfig(
            filename="app.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger() 

    def connect_to_database(self):
        """
        Connects to the InfluxDB database.

        Returns:
            InfluxDBClient: The InfluxDB client instance.
        """
        try:
            if self.client is None or not self.connected:
                self.client = InfluxDBClient(
                    host= self.InfluxIPAddr, port=8086, database="telegraf"
                )
                self.connected = True
            return self.client
        except Exception as e:
            self.connected = False
            self.logger.error(f"Error connecting to the database: {e}")
            return None

    def get_reefer_count(self):
        """
        Retrieve the total count of reefer containers from the simulator configuration file.

        Returns:
            int: The total count of reefer containers.
        """
        dev_data = "/src/simulator_config.yaml"
        with open(dev_data, "r") as c_file:
            try:
                config = yaml.safe_load(c_file)
            except yaml.YAMLError as err:
                err_msg = f"Error while parsing config file: {err}."
                self.logger.error(err_msg)
            return int(config["cargo_info"]["reefer_containers"])

    def get_dew_count(self, client):
        """
        Retrieve the count of containers with dewpoint alerts.

        Returns:
            int: The count of containers with dewpoint alerts.
        """
        query = 'SELECT last("Dewpoint") as last_dewpoint, last("Alert") as last_alert FROM "dewpoint" GROUP BY "container"'
        try:
            result = client.query(query)
        except Exception as e:
            client = None
            self.logger.error(f"Error fetching last value: {e}")
            return None
        
        result = client.query(query)
        # Parse the result and filter containers with last_alert = False
        alert_count=0
        for container_key, container_results in result.items():
            container_id = container_key[1]['container']
            container_result = list(container_results)
            dewpoint = container_result[0]['last_dewpoint']
            alert = container_result[0]['last_alert']

            if alert:
                alert_count+=1
        self.logger.info(alert_count)
        return int(alert_count)

    def create_gauage(self, client):
        """
        Create a gauge chart displaying the current power consumption.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly figure object representing the gauge chart.
        """

        query = f'SELECT * FROM powerconsumption ORDER BY time DESC LIMIT 1'
        try:
            result = client.query(query)
        except Exception as e:
            client = None
            self.logger.error(f"Error fetching last value: {e}")
            # return None
        
        # Query the database
        try:
            result = client.query(query)
            # Extract the last value from the result
            last_power_value = result.raw['series'][0]['values'][0][1]
            self.logger.info(last_power_value)
            result = ((last_power_value) * self.get_reefer_count())
            
        except Exception as e:
            result = 558

        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = result,
            mode = "gauge+number+delta",
            delta = {'reference': 200},
            gauge = {'axis': {'range': [None, 1000]},
                    'steps' : [
                        {'range': [0, 200], 'color': "green"},
                        {'range': [200, 900], 'color': "yellow"},
                        {'range': [900, 1000], 'color': "red"}],
                    'threshold' : {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 490},
                    'bar': {'thickness': 0.5},
                    'bgcolor': 'black', 
                    }
                )
            )


        fig.update_layout(
            paper_bgcolor = "black",
            font = {'color': "white", 'family': "Arial"},
            height=160,
        )

        return fig


    def generate_indicator(self, value):
        """
        Generates an indicator figure based on the given value.

        Args:
            value (float or None): The value to represent in the indicator.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure representing the indicator.
        """
        fig = go.Figure()
        if value is None:
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=0,
                    number=dict(
                        font=dict(size=30), prefix="L", suffix="ADING..."
                    ),
                )
            )
        else:
            x = int(value)
            fig.add_trace(
                go.Indicator(
                    mode="number", value=x, number=dict(font=dict(size=80))
                )
            )
        fig.update_layout(
            paper_bgcolor="black",
            plot_bgcolor="black",
            height=160,
            width=170,
            font_color="#c934eb",
        )
        return fig

    def get_worker_data(self, client):
        """
        Retrieves worker data from the database and creates a plot.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure representing stock data.
        """
        query = 'SELECT "current" FROM "combined" WHERE time >= now() - 1m GROUP BY "zone"'
        try:
            result = client.query(query)
        except Exception as e:
            client = None
            self.logger.error(f"Error fetching last value: {e}")
            return None

        all_stock_data = result.raw["series"]
        dfs = []

        if not all_stock_data: # No data retrieved
            fig = go.Figure()
            fig.update_layout(
                paper_bgcolor="black",
                plot_bgcolor="black",
                height=200,
                font_color="white",
                showlegend=True,
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            return fig

        for stock_data in all_stock_data:
            df = pd.DataFrame(
                stock_data["values"], columns=stock_data["columns"]
            )
            df["time"] = pd.to_datetime(df["time"])
            df.sort_values(by="time", inplace=True)
            df["zone"] = stock_data["tags"]["zone"]
            dfs.append(df)

        final_df = pd.concat(dfs, ignore_index=True)

        if "zone" not in final_df.columns:
            raise ValueError(
                "The 'zone' column is missing in the final dataframe."
            )

        fig = go.Figure()
        color_scale = px.colors.qualitative.Dark24

        for i, zone in enumerate(final_df["zone"].unique()):
            zone_data = final_df[final_df["zone"] == zone]
            fig.add_trace(
                go.Scatter(
                    x=zone_data["time"],
                    y=zone_data["current"],
                    mode="lines",
                    line=dict(
                        color=color_scale[i], shape="spline", smoothing=1
                    ),
                    name=zone,
                )
            )

        fig.update_layout(
            paper_bgcolor="black",
            plot_bgcolor="black",
            height=200,
            font_color="white",
            showlegend=True,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        return fig

    def get_alert_data(self, client):
        """
        Retrieves alert count data from the database and creates a plot.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure representing stock data.
        """
        query = 'SELECT AlertCount FROM alert WHERE time >= now() - 1m'
        try:
            result = client.query(query)
        except Exception as e:
            client = None
            self.logger.error(f"Error fetching last value: {e}")
            return None

        all_stock_data = result.raw["series"]
        dfs = []

        if not all_stock_data:  # No data retrieved
            fig = go.Figure()
            fig.update_layout(
                paper_bgcolor="black",
                plot_bgcolor="black",
                height=190,
                font_color="white",
                showlegend=True,
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            return fig

        for stock_data in all_stock_data:
            df = pd.DataFrame(
                stock_data["values"], columns=stock_data["columns"]
            )
            df["time"] = pd.to_datetime(df["time"])
            df.sort_values(by="time", inplace=True)
            dfs.append(df)

        final_df = pd.concat(dfs, ignore_index=True)

        fig = go.Figure()
        color_scale = px.colors.qualitative.Dark24


        fig.add_trace(
            go.Scatter(
                x=final_df["time"],
                y=final_df["AlertCount"],
                mode="lines",
                line=dict(
                    color=color_scale[0], shape="spline", smoothing=1
                ),
                name="Alert Count",
            )
        )

        fig.update_layout(
            paper_bgcolor="black",
            plot_bgcolor="black",
            height=190,
            font_color="white",
            showlegend=True,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        return fig

    def get_trip_info(self):
        """
        Retrieve voyage details from the configuration file and format them into an HTML table.

        Returns:
            str: HTML string representing the voyage details table.
        """
        dev_data = "/src/config/simulator_config.yaml"
        with open(dev_data, "r") as c_file:
            try:
                config = yaml.safe_load(c_file)
            except yaml.YAMLError as err:
                err_msg = f"Error while parsing config file: {err}."
                self.logger.error(err_msg)
                return None

            data = [
                ["Voyage Number", config["voyage_details"]["voyage_number"]],
                ["Vessel Name", config["voyage_details"]["vessel_name"]],
                ["Departure Port", config["voyage_details"]["departure_port"]],
                ["Arrival Port", config["voyage_details"]["arrival_port"]],
            ]

            table_html = f"<h1>Voyage Details</h1>\n{tabulate(data, tablefmt='html')}"
            return table_html

    def get_sweat_cont_info(self, client):
        """
        Retrieve the latest dewpoint and alert status for each container from the database.

        Returns:
            list: A list of lists containing container IDs and their corresponding dewpoints for containers
            with an active alert.
        """
        query = 'SELECT last("Dewpoint") as last_dewpoint, last("Alert") as last_alert FROM "dewpoint" GROUP BY "container"'
        try:
            result = client.query(query)
        except Exception as e:
            client = None
            self.logger.error(f"Error fetching last value: {e}")
            return None
        
        result = client.query(query)
        dew_data = []

        for container_key, container_results in result.items():
            container_id = container_key[1]['container']
            container_result = list(container_results)
            dewpoint = container_result[0]['last_dewpoint']
            alert = container_result[0]['last_alert']

            if alert:
                dew_data.append([container_id, dewpoint])

        return dew_data


    def get_image_path_sweat(self):
        """
        Determine the image path based on the dew count.

        Returns:
            str: The URL of the image corresponding to the dew count status.
        """
        rand = self.get_dew_count(self.connect_to_database())

        if rand==0:
            return f"http://{self.VisualIPAddr}:5000/static/normal_1.png"
        else:
            return f"http://{self.VisualIPAddr}:5000/static/warning_1.png"

    def get_trip_status(self):
        """
        Retrieve the current status of the voyage and report.

        Returns:
            str: HTML-formatted string containing the voyage and report status along with the completion time.
        """

        voyage_status = "In Progress"
        report_status = "Pending"
        completion_time = "0"

        with open("/src/voyage_text/route_status.txt", 'r') as file:
            voyage_status = file.read()
        with open("/src/voyage_text/report_status.txt", 'r') as file:
            report_status = file.read()
        with open("/src/voyage_text/completion_status.txt", 'r') as file:
            completion_time = file.read()
        if voyage_status=="In Progress":
            voyage_status = f"<h1 style='color: #05f7f3;'>{voyage_status}</h1>"
        else:
            voyage_status = f"<h1 style='color: green;'>{voyage_status}</h1>"

        if report_status=="Pending":
            report_status = f"<h1 style='color: red;'>{report_status}</h1>"
        else:
            report_status = f"<h1 style='color: green;'>{report_status}</h1>"

        return f"<h1>Voyage Status:</h1><h1>{voyage_status}</h1><br/><h1>Report Status:</h1><h1>{report_status}</h1><h3>Last generated at: {completion_time} &nbsp; <a href='http://{self.VisualIPAddr}:5000/static/final_report.md' download='final_report.md'> Download Report </a></h3>"

    def get_report(self):
        """
        Retrieve the content of the final report in Markdown format.

        Returns:
            str: The Markdown content of the final report.
        """
        with open("/src/voyage_text/final_report.md", "r", encoding="utf-8") as file:
            markdown_content = file.read()

        return markdown_content

    def get_refeer_container_count(self):
        """
        Get the count of reefer containers and total containers.

        Returns:
            str: HTML formatted string containing the total number of containers, number of reefer containers, and number of dry containers.
        """
        dev_data = "/src/config/simulator_config.yaml"
        with open(dev_data, "r") as c_file:
            try:
                config = yaml.safe_load(c_file)
            except yaml.YAMLError as err:
                err_msg = f"Error while parsing config file: {err}."
                self.logger.error(err_msg)
                return None

            total_containers = f"<h1 style='color: #c934eb; text-align: center; margin: 0; font-size:3rem;'>{config['cargo_info']['total_containers']}</h1>"
            reefer_containers = str(config['cargo_info']['reefer_containers'])
            dry_containers = str(int(config['cargo_info']['total_containers']) - int(config['cargo_info']['reefer_containers']))

            return f"<h3 style='text-align: center; margin: 0; font-weight: normal;'>Total Containers</h3>{total_containers}<h3 style='text-align: center; margin: 0; font-weight: normal;'>Reefers ┃ Dry &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h3><h1 style='color: #c934eb; text-align: center; margin: 0; font-size:2.5rem;'>{reefer_containers} <span style='color:white; font-weight: normal;'>|</span> {dry_containers}</h1>"

    def get_sweat_alert(self):
        """
        Get the sweat alert status based on the dew count.

        Returns:
            gr.Interface: Gradio Interface object with visibility updated based on the dew count.
        """
        rand = self.get_dew_count(self.connect_to_database())

        if rand==0:
            return gr.update(visible=False)
        else:
            return gr.update(visible=True)