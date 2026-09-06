from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class WeatherClientError(Exception):
    """Raised when Open-Meteo cannot provide the requested data."""


@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    admin1: str | None = None


class OpenMeteoClient:
    def __init__(self, timeout_seconds: int = 12):
        self.timeout_seconds = timeout_seconds

    def geocode(self, location_name: str) -> Location:
        params = urllib.parse.urlencode({"name": location_name, "count": 1, "language": "en", "format": "json"})
        url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
        data = self._get_json(url)
        results = data.get("results") or []
        if not results:
            raise WeatherClientError(f"Could not resolve location: {location_name}")

        first = results[0]
        return Location(
            name=first.get("name", location_name),
            latitude=float(first["latitude"]),
            longitude=float(first["longitude"]),
            country=first.get("country"),
            admin1=first.get("admin1"),
        )

    def current_weather(self, location: Location, time_hint: str | None = None) -> dict[str, Any]:
        current_fields = [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
            "wind_gusts_10m",
        ]
        hourly_fields = current_fields + ["precipitation_probability", "uv_index"]
        params = urllib.parse.urlencode(
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": ",".join(current_fields),
                "hourly": ",".join(hourly_fields),
                "forecast_days": 1,
                "timezone": "auto",
            }
        )
        url = f"https://api.open-meteo.com/v1/forecast?{params}"
        data = self._get_json(url)
        current = data.get("current")
        if not current:
            raise WeatherClientError("Open-Meteo returned no current weather values.")

        hourly = data.get("hourly") or {}
        facts = dict(current)
        selected_hourly = self._nearest_hourly_values(hourly, current.get("time"), time_hint)
        if time_hint and selected_hourly:
            facts.update(selected_hourly)
            facts["time_scope"] = time_hint
        else:
            facts.update(selected_hourly)
            facts["time_scope"] = "current"

        facts["weather_time"] = selected_hourly.get("time", current.get("time"))
        facts["weather_description"] = weather_code_description(facts.get("weather_code"))
        facts["location_name"] = display_location(location)
        facts["latitude"] = location.latitude
        facts["longitude"] = location.longitude
        return facts

    def _get_json(self, url: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise WeatherClientError(f"Open-Meteo request failed: {exc}") from exc

    def _nearest_hourly_values(self, hourly: dict[str, Any], current_time: str | None, time_hint: str | None = None) -> dict[str, Any]:
        times = hourly.get("time") or []
        if not times:
            return {}
        index = self._hourly_index_for_hint(times, current_time, time_hint)

        values: dict[str, Any] = {}
        values["time"] = times[index]
        for key, series in hourly.items():
            if key == "time" or not isinstance(series, list) or index >= len(series):
                continue
            value = series[index]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                values[key] = value
        return values

    def _hourly_index_for_hint(self, times: list[str], current_time: str | None, time_hint: str | None) -> int:
        target_hours = {
            "this_morning": 9,
            "morning": 9,
            "this_afternoon": 15,
            "afternoon": 15,
            "this_evening": 18,
            "evening": 18,
        }
        if time_hint in target_hours and current_time:
            target_prefix = f"{current_time[:10]}T{target_hours[time_hint]:02d}"
            for i, hour in enumerate(times):
                if hour[:13] == target_prefix:
                    return i

        if current_time:
            current_hour = current_time[:13]
            for i, hour in enumerate(times):
                if hour[:13] == current_hour:
                    return i
        return 0


def display_location(location: Location) -> str:
    parts = [location.name, location.admin1, location.country]
    return ", ".join(part for part in parts if part)


def weather_code_description(code: Any) -> str:
    descriptions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "slight snow fall",
        73: "moderate snow fall",
        75: "heavy snow fall",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    try:
        return descriptions.get(int(code), "unknown weather code")
    except (TypeError, ValueError):
        return "unknown weather code"
