# Created by Scalers AI for Dell Inc.

from influxdb import InfluxDBClient
import pandas as pd
import os
import time
import logging

class ContextWorker: 
    def __init__(self):
        """
        Initialize the ContextWorker object.

        This class retrieves worker-related data from InfluxDB, processes the events based on the queried data, and stores the result in a DataFrame.
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
                "current",
                "zone"
            FROM
                "combined"
            WHERE
                time > now() - 3m
            GROUP BY
                "zone"
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
        events = []

        power_detected = False
        for index, row in self.df.iterrows():
            if row["current"] > 0:
                if not power_detected:
                    power_detected = True
                    event = {
                        'time': row['time'],
                        'event': 'maritime worker violation Alert',
                        'number of workers': row['current'],
                        'zone ID': row['zone'],
                        'comment': 'Workers entered violation zone'
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
        
        with open("/src/voyage_text/events/worker.txt", "w") as file:
            file.write("Below is the data with events relates to maritime workers enterinrestricted zones of the ship during a Container ship voyage \n\n")
            file.write("# Events\n\n")
            file.write("| Time | Event | Number of Workers | zone ID | Comment |\n")
            file.write("|------|-------|-------------------|---------|---------|\n")
            for index, row in self.events_df.iterrows():
                file.write(f"| {row['time']} | {row['event']} | {row['number of workers']} | {row['zone ID']}  | {row['comment']} |\n")
