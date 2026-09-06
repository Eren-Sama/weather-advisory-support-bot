from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.intent import extract_intent
from src.memory import update_memory
from src.response import compose_sop_response, location_failure_response, no_sop_response, weather_failure_response
from src.sop_matcher import facts_for_sop, load_sops, select_sop
from src.weather import OpenMeteoClient, WeatherClientError


class BotState(TypedDict, total=False):
    user_message: str
    memory: dict[str, Any]
    intent: dict[str, Any]
    location: Any
    weather: dict[str, Any]
    selected_sop: dict[str, Any]
    matched_sops: list[dict[str, Any]]
    matched_conditions: list[dict[str, Any]]
    facts: dict[str, Any]
    error: str
    final_response: str


def build_graph(weather_client: Any | None = None, sop_path: str = "data/sops.json"):
    client = weather_client or OpenMeteoClient()
    sops = load_sops(sop_path)

    def understand_intent(state: BotState) -> BotState:
        memory = state.get("memory") or {}
        intent = extract_intent(state["user_message"], memory)
        if not intent.get("location"):
            return {"intent": intent, "memory": memory, "error": "No location was found in the message or session memory."}
        return {"intent": intent, "memory": memory}

    def resolve_location(state: BotState) -> BotState:
        location_name = state["intent"].get("location")
        if not location_name:
            return {"error": "No location was found in the message or session memory."}
        try:
            return {"location": client.geocode(location_name), "error": ""}
        except WeatherClientError as exc:
            return {"error": str(exc)}

    def fetch_weather(state: BotState) -> BotState:
        try:
            return {"weather": client.current_weather(state["location"], state["intent"].get("time_hint")), "error": ""}
        except WeatherClientError as exc:
            return {"error": str(exc)}

    def match_sop(state: BotState) -> BotState:
        selected_sop, matched_sops = select_sop(state["intent"], state["weather"], sops)
        return {
            "selected_sop": selected_sop,
            "matched_sops": matched_sops,
            "matched_conditions": selected_sop.get("matched_conditions", []) if selected_sop else [],
            "facts": facts_for_sop(selected_sop, state["weather"]),
        }

    def write_final_response(state: BotState) -> BotState:
        response = compose_sop_response(
            state["user_message"],
            state["intent"],
            state["weather"],
            state["selected_sop"],
            state["facts"],
            state.get("matched_conditions", []),
        )
        memory = update_memory(state.get("memory") or {}, state["intent"], state["weather"], state["selected_sop"])
        return {"final_response": response, "memory": memory}

    def write_location_failure(state: BotState) -> BotState:
        return {"final_response": location_failure_response(state.get("error", "unknown location error"))}

    def write_weather_failure(state: BotState) -> BotState:
        return {"final_response": weather_failure_response(state.get("error", "unknown weather error"))}

    def write_no_sop_response(state: BotState) -> BotState:
        memory = update_memory(state.get("memory") or {}, state["intent"], state["weather"], None)
        return {"final_response": no_sop_response(state["weather"]), "memory": memory}

    def route_after_intent(state: BotState) -> str:
        if not state["intent"].get("location"):
            return "location_failure"
        return "resolve_location"

    def route_after_location(state: BotState) -> str:
        if state.get("error"):
            return "location_failure"
        return "fetch_weather"

    def route_after_weather(state: BotState) -> str:
        if state.get("error"):
            return "weather_failure"
        return "match_sop"

    def route_after_matching(state: BotState) -> str:
        if not state.get("selected_sop"):
            return "no_sop"
        return "final_response"

    graph = StateGraph(BotState)
    graph.add_node("understand_intent", understand_intent)
    graph.add_node("resolve_location", resolve_location)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("match_sop", match_sop)
    graph.add_node("final_response", write_final_response)
    graph.add_node("location_failure", write_location_failure)
    graph.add_node("weather_failure", write_weather_failure)
    graph.add_node("no_sop", write_no_sop_response)

    graph.set_entry_point("understand_intent")
    graph.add_conditional_edges("understand_intent", route_after_intent)
    graph.add_conditional_edges("resolve_location", route_after_location)
    graph.add_conditional_edges("fetch_weather", route_after_weather)
    graph.add_conditional_edges("match_sop", route_after_matching)
    graph.add_edge("final_response", END)
    graph.add_edge("location_failure", END)
    graph.add_edge("weather_failure", END)
    graph.add_edge("no_sop", END)

    return graph.compile()


def ask_bot(user_message: str, memory: dict[str, Any] | None = None, weather_client: Any | None = None) -> dict[str, Any]:
    graph = build_graph(weather_client=weather_client)
    return graph.invoke({"user_message": user_message, "memory": memory or {}})
