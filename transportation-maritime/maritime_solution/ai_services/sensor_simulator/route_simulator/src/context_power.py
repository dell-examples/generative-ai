# Created by Scalers AI for Dell Inc.

from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime
import os
import time
import logging

class ContextPower:
    def __init__(self):
        """
        Initialize the ContextPower object.

        This class retrieves power consumption data from InfluxDB, processes the events, and stores the result in a DataFrame.
        """

        # InfluxDB connection details
        self.IPAddr = os.getenv("SERVER_IP", "localhost")
        self.host = self.IPAddr
        self.port = 8086

        # Initialize logger
        logging.basicConfig(
            filename="app.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()

        # Connect to InfluxDB
        self.client = self.establish_influxDB_connection(max_retries=3)

        # Define the InfluxDB query
        self.query = '''
            SELECT
                time,
                "Power"
            FROM
                "powerconsumption"
            WHERE
                time > now() - 3m
            GROUP BY
                "container"
        '''

        # Execute the query
        self.result = self.client.query(self.query)

        # Convert the result to a Pandas DataFrame
        self.df = pd.DataFrame(self.result.get_points())

        # Get the events
        self.events = self.process_events()

        # Convert the events to a Pandas DataFrame
        self.events_df = pd.DataFrame(self.events)

        self.write_text()

        # Close the InfluxDB connection
        self.client.close()


    def establish_influxDB_connection(self, max_retries=3):
        """
        Establish a connection to InfluxDB.

        Parameters:
        - max_retries (int): The maximum number of retries to establish the connection, defaults to 3.

        Returns:
        InfluxDBClient: An InfluxDB client object for interacting with the database.
        """
        retries = 0
        while retries < max_retries:
            try:
                client = InfluxDBClient(self.host, self.port, database="telegraf")
                return client
            except Exception:
                if retries < max_retries - 1:
                    self.logger.error(f"Retrying to get the InfluxDB connection ({retries + 1}/{max_retries})")
                retries += 1
                if retries < max_retries:
                    time.sleep(5)


    # Define a function to process events
    def process_events(self):
        """
        Process power consumption events based on the queried data.

        Returns a list of dictionaries, each representing an event, including the time,
        event type, power consumption, and comments.

        Returns:
        list: A list of dictionaries representing processed events.
        """
        events = []

        # getting max power consumption
        max_row = self.df.loc[self.df["Power"].idxmax()]
        event = {
            'time': datetime.fromisoformat(max_row['time'].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"),
            'event': 'Max Power Consumption',
            'power': f"{round(max_row['Power'], 2)*300} kwh",
            'comment': 'Maximum power consumption of the whole voyage'
        }
        events.append(event)

        # getting min power consumption
        min_row = self.df.loc[self.df["Power"].idxmin()]
        event = {
            'time': datetime.fromisoformat(min_row['time'].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"),
            'event': 'Min Power Consumption',
            'power': f"{round(min_row['Power'], 2)*300} kwh",
            'comment': 'Minimum power consumption of the whole voyage'
        }
        events.append(event)

        # getting all the readings where power consumption has excedded the power consumption limit
        filtered_rows = self.df[self.df["Power"]*300 >= 800.0]
        power_detected = False
        for index, row in self.df.iterrows():
            if row["Power"]*300 >= 800.0:
                if not power_detected:
                    power_detected = True
                    event = {
                        'time': row['time'],
                        'event': 'Power Consumption Alert',
                        'power': f"{row['Power']*300} kwh",
                        'comment': 'Power consumption crossed threshold value'
                    }
                    events.append(event)
            else:
                power_detected = False
        return events

    def write_text(self):
        """
        Write processed events data to a text file.

        Returns:
        None
        """

        if not os.path.exists("/src/voyage_text/events"):
            os.makedirs("/src/voyage_text/events")
        
        with open("/src/voyage_text/events/power.txt", "w") as file:
            file.write("Below is the data with events relates to how is the power consumption pattern of the ship during a Container ship voyage \n\n")
            file.write("# Events\n\n")
            file.write("| Time | Event | Power Consumption | Comment |\n")
            file.write("|------|-------|-------------------|---------|\n")
            for index, row in self.events_df.iterrows():
                file.write(f"| {row['time']} | {row['event']} | {row['power']} | {row['comment']} |\n")
