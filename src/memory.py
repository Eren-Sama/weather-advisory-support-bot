from __future__ import annotations

from typing import Any


def update_memory(memory: dict[str, Any], intent: dict[str, Any], weather: dict[str, Any] | None, sop: dict[str, Any] | None) -> dict[str, Any]:
    updated = dict(memory or {})
    if intent.get("location"):
        updated["last_location"] = intent["location"]
    if "activity" in intent and intent.get("activity") is None:
        updated.pop("last_activity", None)
    elif intent.get("activity"):
        updated["last_activity"] = intent["activity"]
    if intent.get("category"):
        updated["last_category"] = intent["category"]
    if intent.get("question_type"):
        updated["last_question_type"] = intent["question_type"]
    if "vulnerable_group" in intent and intent.get("vulnerable_group") is None:
        updated.pop("last_vulnerable_group", None)
    elif intent.get("vulnerable_group"):
        updated["last_vulnerable_group"] = intent["vulnerable_group"]
    if weather:
        updated["last_weather"] = {
            "location_name": weather.get("location_name"),
            "time": weather.get("time"),
            "temperature_2m": weather.get("temperature_2m"),
            "precipitation": weather.get("precipitation"),
            "wind_speed_10m": weather.get("wind_speed_10m"),
        }
    if sop:
        updated["last_sop_id"] = sop.get("id")
    return updated
