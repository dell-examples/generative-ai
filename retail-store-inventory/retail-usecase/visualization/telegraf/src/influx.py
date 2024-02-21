# Created by Scalers AI for Dell Inc.

from influxdb import InfluxDBClient
from requests.exceptions import ConnectionError
import time
import os

# define the details for connecting to InfluxDB
host = os.getenv("HOSTNAME")  # replace with your InfluxDB host
port = os.getenv("PORT")  # replace with your InfluxDB port if different
retry_limit = 5  # replace with the number of times you want to retry

# function to check if InfluxDB is running
def check_if_influxdb_is_running(client, retry_limit=5, delay=1):
    for try_count in range(retry_limit):
        try:
            client.ping()
            return True
        except ConnectionError:
            print(f'InfluxDB not accessible, retrying in {delay} seconds...')
            time.sleep(delay)
            delay *= 2
    return False

# Initialize the InfluxDB client
client = InfluxDBClient(host=host, port=port)

# Check whether InfluxDB is running or not
if not check_if_influxdb_is_running(client, retry_limit):
    print("InfluxDB is not running after multiple retry attempts. Please start it and run the script again.")
    exit(1)

# Create a database
database_name = os.getenv("DB_NAME")  # replace with your database name
client.create_database(database_name)

# Switch to the newly created database
client.switch_database(database_name)

# Create a retention policy
policy_name = os.getenv("POLICY_NAME")  # replace with your retention policy name
duration = os.getenv("DURATION")  # replace with your duration. format: 'time_number time_unit' e.g., '30d' means retain data for 30 days
replication = '1'  # replication factor
default = True  # set this policy as default

client.create_retention_policy(policy_name, duration, replication, database_name, default=default)