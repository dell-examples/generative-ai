# Created by Scalers AI for Dell Inc.

import openmeteo_requests
import requests_cache
from retry_requests import retry

class OpenMeteo:
    def __init__(self):
        """
        Initialize an OpenMeteo client.

        Returns:
        None
        """
        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        self.retry_session = retry(self.cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = self.retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        self.url = "https://archive-api.open-meteo.com/v1/archive"

    def get_vals(self, lat, long, date, hour):
        """
        Get weather values for a specific location, date, and hour.

        Args:
        lat (float): Latitude of the location.
        long (float): Longitude of the location.
        date (str): Date in 'YYYY-MM-DD' format.
        hour (int): Hour of the day (0-23).

        Returns:
        Tuple[float, float]: A tuple containing the temperature (in Celsius) and relative humidity
        (in percentage) for the specified location, date, and hour.
        """
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": str(date),
            "end_date": str(date),
            "hourly": ["temperature_2m", "relative_humidity_2m"]
        }
        responses = self.openmeteo.weather_api(self.url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        return hourly_temperature_2m[hour], hourly_relative_humidity_2m[hour]
    
    def normalize_latitude(self, latitude):
        """
        Normalize latitude value to be within the range [-90, 90].

        Args:
        latitude (float): Latitude value to be normalized.

        Returns:
        float: Normalized latitude value within the range [-90, 90].
        """
        normalized_latitude = max(min(latitude, 90), -90)
        return normalized_latitude

    def normalize_longitude(self, longitude):
        """
        Normalize longitude value to be within the range [-180, 180].

        Args:
        longitude (float): Longitude value to be normalized.

        Returns:
        float: Normalized longitude value within the range [-180, 180].
        """
        normalized_longitude = (longitude + 180) % 360 - 180
        return normalized_longitude
    
    def get_metrics(self, lat, long, time):
        """
        Get weather metrics for a specific location and time.

        Args:
        lat (float): Latitude of the location.
        long (float): Longitude of the location.
        time (datetime.datetime): Date and time for which weather metrics are requested.

        Returns:
        Tuple[float, float]: A tuple containing the temperature (in Celsius) and relative humidity
        (in percentage) for the specified location and time.
        """
        lat = self.normalize_latitude(lat)
        long = self.normalize_longitude(long)
        return self.get_vals(lat, long, str(time.date()), time.hour)
