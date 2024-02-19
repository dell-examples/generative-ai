# Created by Scalers AI for Dell Inc.

from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime
import os
import time
import logging

class ContextSweat:
    def __init__(self):
        """
        Initialize the ContextSweat object.

        This class retrieves dewpoint data from InfluxDB, processes the events where alerts are raised, and stores the result in a DataFrame.
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
                 COUNT(DISTINCT("ContainerID")),
                 MEAN("Dewpoint") AS "average_dewpoint"
             FROM
             "dewpoint"
             WHERE
             time > now() - 3m AND "Alert" = true
             GROUP BY
                 time(5s)
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
        Process dewpoint events based on the queried data.

        Returns a list of dictionaries, each representing an event, including the time, event type, number of containers affected, average dewpoint, and comments.

        Returns:
        list: A list of dictionaries representing processed dewpoint events.
        """

        events = []
        sweating_detected = False

        for index, row in self.df.iterrows():
            if row['count'] > 0:
                if not sweating_detected:
                    sweating_detected = True
                    event = {
                        'time': datetime.fromisoformat(row['time'].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"),
                        'event': 'Sweating Detected',
                        'container_count': row['count'],
                        'dewpoint': round(row['average_dewpoint'], 2),
                        'comment': 'Sweating observed, analytics triggered'
                    }
                    events.append(event)
            elif sweating_detected:
                sweating_detected = False
                event = {
                    'time': datetime.fromisoformat(row['time'].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"),
                    'event': 'Sweating Resolved',
                    'container_count': row['count'],
                    'dewpoint': round(row['average_dewpoint'], 2),
                    'comment': 'Sweating resolved, ventilators deactivated'
                }
                events.append(event)
        return events


    def write_text(self):
        """
        Write processed sweating events data to a text file.

        Returns:
        None
        """      
        if not os.path.exists("/src/voyage_text/events"):
            os.makedirs("/src/voyage_text/events")
        
        with open("/src/voyage_text/events/sweat.txt", "w") as file:
            file.write("Below is the data with events relates to how the container sweating is managed with ventilator systems during a  Container ship voyage \n\n")
            file.write("# Events\n\n")
            file.write("| Time | Event | Number of Containers  | Dewpoint | Comment |\n")
            file.write("|------|-------|--------------|----------|---------|\n")
            for index, row in self.events_df.iterrows():
                file.write(f"| {row['time']} | {row['event']} | {row['container_count']} | {row['dewpoint']} | {row['comment']} |\n")
