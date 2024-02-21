# Created by Scalers AI for Dell Inc.

import os

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from influxdb import InfluxDBClient
from yaml.loader import SafeLoader


class RetailInventoryDashboard:
    """
    A class to create a Retail Inventory Dashboard using Gradio and InfluxDB.
    """

    def __init__(self):
        """
        Initializes the RetailInventoryDashboard object.
        """
        self.IPAddr = os.getenv("SERVER_IP", "localhost")
        self.client = None
        self.connected = False

    def connect_to_database(self):
        """
        Connects to the InfluxDB database.

        Returns:
            InfluxDBClient: The InfluxDB client instance.
        """
        try:
            if self.client is None or not self.connected:
                self.client = InfluxDBClient(
                    host="localhost", port=8086, database="telegraf"
                )
                self.connected = True
            return self.client
        except Exception as e:
            self.connected = False
            print(f"Error connecting to the database: {e}")
            return None

    def get_last_value(self, client, field, measurement):
        """
        Retrieves the last value for a specific field in a measurement.

        Args:
            client (InfluxDBClient): The InfluxDB client instance.
            field (str): The field to retrieve the last value from.
            measurement (str): The measurement to query.

        Returns:
            float or None: The last value for the field in the measurement.
        """
        try:
            query = f'SELECT last("{field}") FROM "{measurement}" WHERE time >= now() - 1m'
            results = client.query(query)
            last_value = None

            for point in results.get_points():
                last_value = point["last"]
            return last_value
        except Exception as e:
            self.client = None
            print(f"Error fetching last value: {e}")
            return None

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
                    mode="number", value=x, number=dict(font=dict(size=50))
                )
            )
        fig.update_layout(
            paper_bgcolor="black",
            plot_bgcolor="black",
            height=130,
            width=290,
            font_color="#c934eb",
        )
        return fig

    def get_stock_data(self):
        """
        Retrieves stock data from the database and creates a plot.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure representing stock data.
        """
        query = 'SELECT "current" FROM "retail" WHERE time >= now() - 1m GROUP BY "zone"'
        try:
            result = self.client.query(query)
        except Exception as e:
            self.client = None
            print(f"Error fetching last value: {e}")
            return None

        all_stock_data = result.raw["series"]
        dfs = []

        if not all_stock_data:  # No data retrieved
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

    def get_device_info(self):
        dev_data = "/src/config/config.yaml"
        with open(dev_data, "r") as c_file:
            try:
                config = yaml.safe_load(c_file)
            except yaml.YAMLError as err:
                err_msg = f"Error while parsing config file: {err}."
                print(err_msg)
            data = [
                ["Device", config["product_name"]],
                ["CPU", config["CPU_name"]],
                ["SOCKETS", config["num_sockets"]],
                ["Number Of Cores", config["num_cores"]],
            ]
            return data

    def create_dashboard(self):
        """
        Creates the Retail Inventory Management dashboard using Gradio.
        """
        with gr.Blocks(
            title="Retail Inventory Management", css="website.css"
        ) as demo:
            with gr.Row(elem_id="row2"):
                with gr.Column(scale=4, elem_id="col1"):
                    with gr.Box(elem_id="stream"):
                        iframe = gr.HTML(
                            value=f"<iframe src='http://{self.IPAddr}:9000/' width='1150px' height='600px'></iframe>",
                        )
                with gr.Column(scale=1, elem_id="col2"):
                    with gr.Box(elem_id="box2"):
                        with gr.Column():
                            with gr.Row():
                                curr_items = gr.Plot(
                                    label="Current Items",
                                    show_label=True,
                                    elem_id="curr_items",
                                )
                            with gr.Row():
                                total_items = gr.Plot(
                                    label="Total Items",
                                    show_label=True,
                                    elem_id="total_items",
                                )
                            with gr.Row():
                                info = gr.Dataframe(
                                    headers=None,
                                    wrap=True,
                                    interactive=False,
                                    row_count=(4, "fixed"),
                                    col_count=(2, "fixed"),
                                    value=self.get_device_info,
                                    every=5,
                                    elem_id="info",
                                )
            with gr.Row(elem_id="row3"):
                with gr.Box(elem_id="box3"):
                    stock = gr.Plot(
                        label="Current Stock",
                        show_label=False,
                        elem_id="stock",
                    )

            demo.load(self.get_stock_data, inputs=[], outputs=[stock], every=5)
            demo.load(
                lambda: self.generate_indicator(
                    self.get_last_value(
                        self.connect_to_database(), "current", "combined"
                    )
                ),
                inputs=[],
                outputs=[curr_items],
                every=1,
            )
            demo.load(
                lambda: self.generate_indicator(
                    self.get_last_value(
                        self.connect_to_database(), "total", "combined"
                    )
                ),
                inputs=[],
                outputs=[total_items],
                every=1,
            )
        return demo

    def run(self):
        """
        Runs the Retail Inventory Dashboard.
        """
        demo = self.create_dashboard()
        demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    dashboard = RetailInventoryDashboard()
    dashboard.run()
