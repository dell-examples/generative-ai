# Created by Scalers AI for Dell Inc.

import os
import time

from influxdb import InfluxDBClient
from requests.exceptions import ConnectionError


def check_if_influxdb_is_running(client, retry_limit=5, delay=1):
    """Retry connecting to Influx DB until a retry limit."""
    for _ in range(retry_limit):
        try:
            client.ping()
            return True
        except ConnectionError:
            print(f"InfluxDB not accessible, retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
    return False


def create_database():
    """Create database on the influxdb.
    This script is to handle the long wait time by telegraf
    on creating the database.
    """
    host = os.getenv("INFLUXDB_HOSTNAME", "localhost")
    port = os.getenv("INFLUXDB_PORT", 8086)
    database_name = os.getenv("INFLUXDB_DATABASE_NAME", "factory")
    retry_limit = 5  # Number of times to retry

    # Initialize the InfluxDB client
    client = InfluxDBClient(host=host, port=port)

    # Check if InfluxDB is running
    if not check_if_influxdb_is_running(client, retry_limit):
        print("InfluxDB is not running after multiple retry attempts.")
        exit(1)

    # Create a database
    client.create_database(database_name)

    # Switch to the newly created database
    client.switch_database(database_name)


if __name__ == "__main__":
    create_database()
