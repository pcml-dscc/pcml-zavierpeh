# MLFP06 — Task 1: Schema-Constrained Extraction

**Weight**: 20 marks · **Difficulty**: Hard · **Framework**: Kaizen
`Signature` + `BaseAgent` (Ollama, `llama3.2:3b`) · **Dataset**: 6 fixed SG
logistics incident reports (given in the starter)

## Scenario

A Singapore last-mile logistics operator receives free-text incident reports
from drivers and depot staff. The downstream insurance + ops pipeline is
strongly typed — it needs one clean, validated record per report or the row
insert fails. Free-form JSON prompting drifts (wrong key names, code fences,
single quotes). The production fix is a **typed Kaizen Signature**: you declare
the schema in Python, Kaizen renders it into the prompt and validates the
response, and you get a typed dict back.

Drive the local Ollama LLM at **temperature 0** (deterministic) and extract a
structured record from each of the six reports.

Implement `solve() -> list[dict]`.

## Required schema (every record, exactly these five keys)

| Field              | Type   | Meaning                                                |
| ------------------ | ------ | ------------------------------------------------------ |
| `incident_id`      | `str`  | The incident reference id (e.g. `INC-3001`)            |
| `severity`         | `str`  | Exactly one of `low`, `medium`, `high`                 |
| `location`         | `str`  | The facility/location named in the report              |
| `parcels_affected` | `int`  | Number of parcels affected                             |
| `claim_required`   | `bool` | `True` if an insurance claim is required, else `False` |

## What to build

1. **Define the Signature** — `IncidentExtraction(Signature)` with one
   `InputField` (`report_text: str`) and five `OutputField`s matching the table
   above. The `OutputField` descriptions steer the LLM — be precise.
2. **Build the agent** — a `BaseAgent` subclass backed by your Signature, wired
   to Ollama: `config={"model": DEFAULT_CHAT_MODEL, "llm_provider": "ollama",
"base_url": OLLAMA_BASE_URL, "use_async_llm": True, "temperature": 0.0}`.
3. **Extract** — run `await agent.run_async(report_text=report)` for each of the
   six reports and collect the five fields per record.

## Visible sanity check

The first report (`INC-3001`, Tuas Checkpoint, 42 parcels, claim required)
should extract to roughly:

```python
{"incident_id": "INC-3001", "severity": "high", "location": "Tuas Checkpoint",
 "parcels_affected": 42, "claim_required": True}
```

(severity casing may vary — it is graded case-insensitively.)

## Grading (11 automated checks, all must pass)

return type is list · 6 records · all items are dicts · all five schema keys
present in every record · types correct (`incident_id` present, `parcels_affected`
→ int, `claim_required` → bool) · `severity` in `{low,medium,high}` ·
`incident_id` exact for all 6 · `severity` correct (≥5/6) · `location` correct
(≥5/6) · `parcels_affected` correct (≥5/6) · `claim_required` correct (≥5/6).

**Why the 5/6 floor?** The LLM runs at temperature 0 (greedy, deterministic), so
extraction is stable. The semantic-field checks still use a 5-of-6 floor to
tolerate at most one occasional drift; the explicit `incident_id` is graded
exactly because it is copied verbatim from the text. Schema and type checks are
fully deterministic.

## Rules

- **Kaizen Signature only** — no hand-rolled `json.loads` of free-form text.
- **Local Ollama only** — no cloud models, no API keys. Temperature 0.
- Do not mutate `INCIDENT_REPORTS`.
