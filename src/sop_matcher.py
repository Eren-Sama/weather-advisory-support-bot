from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SEVERITY_RANK = {"low": 1, "moderate": 2, "high": 3, "critical": 4}


def load_sops(path: str | Path = "data/sops.json") -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def select_sop(intent: dict[str, Any], weather: dict[str, Any], sops: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    matches = []
    for sop in sops:
        matched_conditions = matched_sop_conditions(sop, intent, weather)
        if matched_conditions is not None:
            match = dict(sop)
            match["matched_conditions"] = matched_conditions
            matches.append(match)
    matches.sort(key=lambda sop: (SEVERITY_RANK.get(sop.get("severity", "low"), 0), sop.get("priority", 0)), reverse=True)
    return (matches[0] if matches else None), matches


def sop_matches(sop: dict[str, Any], intent: dict[str, Any], weather: dict[str, Any]) -> bool:
    return matched_sop_conditions(sop, intent, weather) is not None


def matched_sop_conditions(sop: dict[str, Any], intent: dict[str, Any], weather: dict[str, Any]) -> list[dict[str, Any]] | None:
    condition_block = sop.get("conditions", {})
    all_conditions = condition_block.get("all", [])
    any_conditions = condition_block.get("any", [])

    matched_all = []
    for condition in all_conditions:
        result = evaluate_condition(condition, intent, weather)
        if not result["matched"]:
            return None
        matched_all.append(result)

    matched_any = []
    for condition in any_conditions:
        result = evaluate_condition(condition, intent, weather)
        if result["matched"]:
            matched_any.append(result)

    if any_conditions and not matched_any:
        return None
    if not all_conditions and not any_conditions:
        return None
    return matched_all + matched_any


def evaluate_condition(condition: dict[str, Any], intent: dict[str, Any], weather: dict[str, Any]) -> dict[str, Any]:
    source = condition.get("source")
    field = condition.get("field")
    operator = condition.get("operator")
    expected = condition.get("value")
    actual = intent.get(field) if source == "intent" else weather.get(field)
    result = {
        "source": source,
        "field": field,
        "operator": operator,
        "expected": expected,
        "actual": actual,
        "matched": False,
    }

    if operator == "equals":
        result["matched"] = actual == expected
        return result
    if operator == "in":
        if isinstance(actual, list):
            result["matched"] = any(item in expected for item in actual)
            return result
        result["matched"] = actual in expected
        return result
    if operator == "not_in":
        result["matched"] = actual not in expected
        return result
    if operator == "contains_any":
        if actual is None:
            return result
        result["matched"] = any(str(item).lower() in str(actual).lower() for item in expected)
        return result

    if actual is None:
        return result

    try:
        actual_number = float(actual)
        expected_number = float(expected)
    except (TypeError, ValueError):
        return result

    if operator == "gte":
        result["matched"] = actual_number >= expected_number
        return result
    if operator == "lte":
        result["matched"] = actual_number <= expected_number
        return result
    if operator == "gt":
        result["matched"] = actual_number > expected_number
        return result
    if operator == "lt":
        result["matched"] = actual_number < expected_number
        return result

    return result


def facts_for_sop(sop: dict[str, Any] | None, weather: dict[str, Any]) -> dict[str, Any]:
    if not sop:
        return {}
    return {key: weather.get(key) for key in sop.get("facts_to_mention", []) if key in weather and weather.get(key) is not None}
