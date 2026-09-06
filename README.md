# Weather-Advisory Support Bot

This project implements the BrainWave internship assignment: a small LangGraph chatbot that answers outdoor safety questions using live Open-Meteo weather and editable SOP rules.

## What It Does

1. Reads the user's outdoor activity question.
2. Extracts simple intent, including location, activity, category, vulnerable group, and whether the user is asking a safety or comfort question.
3. Resolves the location with Open-Meteo geocoding.
4. Fetches live weather from Open-Meteo forecast. For follow-ups such as "this evening" or "this afternoon", it uses the matching hourly forecast slice for today.
5. Loads SOPs from `data/sops.json`.
6. Selects the highest-severity matching SOP using deterministic rule checks, with priority used as the tie-breaker.
7. Uses Groq only for intent extraction and optional wording polish when an API key is available. Weather facts and SOP choice are controlled by code.

If location lookup, weather fetching, or SOP matching fails, the bot says that honestly instead of guessing.

## Folder Structure

```text
weather-advisory-support-bot/
  app.py
  evals.py
  requirements.txt
  .env.example
  README.md
  data/
    sops.json
  src/
    graph.py
    intent.py
    llm.py
    memory.py
    response.py
    sop_matcher.py
    weather.py
```

## Architecture

The graph in `src/graph.py` is a real LangGraph:

```text
understand_intent
  -> location_failure OR resolve_location
resolve_location
  -> location_failure OR fetch_weather
fetch_weather
  -> weather_failure OR match_sop
match_sop
  -> no_sop OR final_response
```

The failure paths are separate nodes so the reviewer can see exactly where the bot stops and why.

## Design Principle

The LLM helps with language, not policy decisions.

Intent extraction can use Groq when available, but weather values come from Open-Meteo and SOP matching is deterministic. This keeps the safety decision traceable and makes policy changes possible through `data/sops.json` without changing the graph.

## SOP Design

SOPs live in `data/sops.json`. Each SOP has:

- `id`
- `title`
- `category`
- `severity`
- `priority`
- `conditions`
- `guidance`
- `facts_to_mention`

The matcher supports generic operators such as `equals`, `in`, `gte`, `lte`, `gt`, `lt`, `not_in`, and `contains_any`. This keeps individual SOP logic out of the graph and out of Python control flow.

The included SOPs cover:

- outdoor exercise
- travel and commute
- vulnerable groups
- leisure and picnic comfort
- general severe weather

The current policy file contains 13 SOPs across four severity levels: low, moderate, high, and critical. `SOP-PICNIC-COMFORT-001` is intentionally a comfort-focused rule rather than a numeric weather-threshold rule: it handles "nice day for a picnic/park/outdoor lunch" style questions as a comfort check, and stronger safety SOPs outrank it when dangerous weather is present. `SOP-EXERCISE-FAIR-001` covers ordinary outdoor exercise only when no stronger risk SOP outranks it, so common cycling/running questions still get a traceable policy answer.

When more than one SOP matches, the bot chooses the highest severity, then highest priority. I chose this because it is easy to explain and safer than returning a low-severity comfort answer when a high-severity safety rule also matches.

## Weather Grounding

Weather data comes from Open-Meteo:

- geocoding: `https://geocoding-api.open-meteo.com/v1/search`
- forecast: `https://api.open-meteo.com/v1/forecast`

The forecast request explicitly asks for current fields and hourly fields. For normal "today/current" questions, hourly `uv_index` and `precipitation_probability` are copied from the nearest current hour. For "this morning", "this afternoon", and "this evening", the bot uses the matching hourly forecast slice and then re-runs SOP matching.

Groq is not used as the source of weather facts. Final answers are first composed deterministically from:

- selected SOP
- exact weather facts returned by Open-Meteo
- exact SOP condition that triggered the match, such as `precipitation_probability >= 60`

When Groq is available, it can polish that deterministic answer for readability. The code only accepts the polished answer if it still includes the selected SOP, preserves the required fact values, avoids unapproved numbers, and does not add unsupported hazards such as flooding or official alerts. Otherwise, the deterministic answer is returned.

The LLM intent output is also validated before use. Enum-like fields such as `category`, `activity`, `question_type`, `time_hint`, and `vulnerable_group` must be from allowed values. Unexpected values are ignored so the rule-based intent fallback can still drive SOP matching. Evals run with `USE_LLM=0` so the policy engine is tested deterministically even if a Groq key exists.

## Setup

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file if you want Groq intent extraction and nicer final wording:

```text
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

You can copy `.env.example` to `.env` and paste your real Groq key there. Keep `.env` private; it is ignored by git.

`GROQ_MODEL` is optional. The default is `llama-3.1-8b-instant`, which is a fast Groq-hosted model suitable for this assignment. Open-Meteo does not need an API key.

Without a Groq API key, the project still runs with simple rule-based intent extraction and deterministic response wording. That fallback is mainly for local testing and review when no LLM key is available.

## Run The Chat UI

```bash
streamlit run app.py
```

Try:

```text
Is it safe to cycle in Bhopal today?
What about this evening instead?
Can I take my kid to the park in Jaipur?
Does it look good for a picnic in Indore?
```

The Streamlit session stores lightweight memory, so follow-up questions can reuse the previous location, activity, and category.

## Run Evals

```bash
python evals.py
```

The eval suite covers:

- direct deterministic policy-engine tests for SOP thresholds
- direct multi-match policy tests for SOP priority and severity ranking
- deterministic intent fallback for scooter/two-wheeler wording
- validation of invalid LLM intent values
- rejection of unsafe LLM polishing that adds unsupported hazards
- direct SOP match for cycling wind
- direct SOP match for vulnerable-group heat
- paraphrased travel/rain question
- rain-probability grounding when current precipitation is 0 mm
- paraphrased fuzzy picnic comfort question
- routine low-risk outdoor exercise
- no matching SOP
- simulated weather API failure
- adversarial prompt-injection attempt
- conversational "should I postpone?" follow-up
- "this evening" time-change follow-up that re-runs SOP matching
- intent-change follow-up from commute to outdoor lunch
- memory reset when the user says to forget the specific activity

There is also an optional live Open-Meteo check:

```bash
set RUN_LIVE_EVAL=1
python evals.py
```

The deterministic fake-weather tests are the correctness tests. The optional live check is only an integration smoke test because real weather changes. It is useful for confirming that Open-Meteo integration works and that the bot grounds its answer in current API facts, but it should not be treated as a stable policy-correctness test.

## Adding Another SOP

Add a new object to `data/sops.json`. For example:

```json
{
  "id": "SOP-FOG-TRAVEL-001",
  "title": "Fog risk for morning travel",
  "category": "travel",
  "severity": "moderate",
  "priority": 58,
  "conditions": {
    "all": [
      {"source": "intent", "field": "category", "operator": "in", "value": ["travel", "commute"]}
    ],
    "any": [
      {"source": "weather", "field": "weather_code", "operator": "in", "value": [45, 48]}
    ]
  },
  "guidance": "Warn about low visibility and recommend delaying travel or using extra caution.",
  "facts_to_mention": ["weather_code", "weather_description"]
}
```

No graph, weather, or LLM code changes are needed because the matcher evaluates generic conditions from the JSON file.

## Decisions And Limitations

- City disambiguation is simple: the first Open-Meteo geocoding result is used. A production system should ask the user to choose when there are several likely matches.
- The bot handles simple same-day time hints: this morning, this afternoon, and this evening. It does not yet support arbitrary future dates or detailed time ranges.
- The SOP matcher is intentionally understandable, not an ML classifier. This is good for review and policy traceability, but it may miss unusual phrasing.
- The system does not include official weather alerts such as IMD warnings. It only uses Open-Meteo weather fields, so broad synoptic events must be represented indirectly through available live values unless an alert source is added later.
- This is an internship assignment implementation, not a production safety product.
