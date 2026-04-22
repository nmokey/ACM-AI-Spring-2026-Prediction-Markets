"""
data/ingestion/weather_client.py
──────────────────────────────────
NOAA Weather API client.

Team 1 — implement all methods marked with TODO.

Docs:    https://www.weather.gov/documentation/services-web-api
Auth:    None required — completely free, no API key needed.
Base URL: https://api.weather.gov

Useful endpoints:
    GET /points/{lat},{lon}           → returns gridId, gridX, gridY for a location
    GET /gridpoints/{office}/{x},{y}/forecast/hourly  → hourly forecast periods
"""

from __future__ import annotations

import requests
from typing import Any

BASE_URL = "https://api.weather.gov"

# NOAA requires a User-Agent header identifying your application.
# Without it, requests will be rejected.
HEADERS = {"User-Agent": "ACM-AI-PredictionMarkets/1.0 (UCLA student project)"}

# Grid points for target cities. To find the gridpoint for a new city:
#   GET https://api.weather.gov/points/{lat},{lon}
# and read .properties.gridId, .properties.gridX, .properties.gridY
CITY_GRIDPOINTS: dict[str, dict[str, Any]] = {
    "New York": {"office": "OKX", "gridX": 33, "gridY": 37},
    "Los Angeles": {"office": "LOX", "gridX": 149, "gridY": 48},
    "Chicago": {"office": "LOT", "gridX": 76, "gridY": 73},
}


class WeatherClient:

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def get_forecast(self, city: str) -> list[dict[str, Any]]:
        """
        Fetch the hourly forecast periods for a city.

        Args:
            city: must be a key in CITY_GRIDPOINTS (e.g. "New York")

        Returns:
            List of forecast period dicts from NOAA. Each period contains:
            startTime, endTime, temperature, temperatureUnit,
            probabilityOfPrecipitation, shortForecast, etc.

        TODO (Week 2):
            - Look up the city in CITY_GRIDPOINTS (raise ValueError if not found)
            - Build the forecast URL using office, gridX, gridY
            - Make a GET request with self.session
            - Return resp.json()["properties"]["periods"]
        """
        raise NotImplementedError

    def get_todays_precip_prob(self, city: str) -> float | None:
        """
        Return the maximum precipitation probability (0–100) across today's forecast hours.
        Returns None if the data is unavailable.

        TODO (Week 2):
            - Call self.get_forecast(city)
            - Filter periods to only today's date (check startTime)
            - Extract probabilityOfPrecipitation["value"] from each period
            - Return the max, or None if all values are None
        """
        raise NotImplementedError
        


# ── Week 1 hello world ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TODO (Week 1): make a raw requests.get() call to the NOAA hourly forecast
    # endpoint for one of the cities in CITY_GRIDPOINTS and print today's
    # precipitation probability. Push your notebook to notebooks/week1_team1.ipynb.
    print("Hello from NOAA! Implement me.")
